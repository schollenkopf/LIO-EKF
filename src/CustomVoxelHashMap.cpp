// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "CustomVoxelHashMap.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>
#include <random>    // for std::shuffle
#include <algorithm> // for std::shuffle

// This parameters are not intended to be changed, therefore we do not expose it
namespace
{
    struct ResultTuple
    {
        ResultTuple(std::size_t n)
        {
            source.reserve(n);
            target.reserve(n);
        }
        std::vector<Eigen::Vector3d> source;
        std::vector<Eigen::Vector3d> target;
    };
} // namespace

namespace kiss_icp
{

    CustomVoxelHashMap::Vector3dVectorTuple CustomVoxelHashMap::GetCorrespondences(
        const Vector3dVector &points, double max_correspondance_distance) const
    {
        // Lambda Function to obtain the KNN of one point, maybe refactor
        auto GetClosestNeighboor = [&](const Eigen::Vector3d &point)
        {
            auto kx = static_cast<int>(point[0] / voxel_size_);
            auto ky = static_cast<int>(point[1] / voxel_size_);
            auto kz = static_cast<int>(point[2] / voxel_size_);
            std::vector<Voxel> voxels;
            voxels.reserve(27);
            for (int i = kx - 1; i < kx + 1 + 1; ++i)
            {
                for (int j = ky - 1; j < ky + 1 + 1; ++j)
                {
                    for (int k = kz - 1; k < kz + 1 + 1; ++k)
                    {
                        voxels.emplace_back(i, j, k);
                    }
                }
            }

            using Vector3dVector = std::vector<Eigen::Vector3d>;
            Vector3dVector neighboors;
            neighboors.reserve(27 * max_points_per_voxel_);
            std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &voxel)
                          {
            auto search = map_.find(voxel);
            if (search != map_.end()) {
                const auto &points = search->second.points;
                if (!points.empty()) {
                    for (const auto &point : points) {
                        neighboors.emplace_back(point);
                    }
                }
            } });

            Eigen::Vector3d closest_neighbor;
            double closest_distance2 = std::numeric_limits<double>::max();
            std::for_each(neighboors.cbegin(), neighboors.cend(), [&](const auto &neighbor)
                          {
            double distance = (neighbor - point).squaredNorm();
            if (distance < closest_distance2) {
                closest_neighbor = neighbor;
                closest_distance2 = distance;
            } });

            return closest_neighbor;
        };
        using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
        const auto [source, target] = tbb::parallel_reduce(
            // Range
            tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
            // Identity
            ResultTuple(points.size()),
            // 1st lambda: Parallel computation
            [max_correspondance_distance, &GetClosestNeighboor](
                const tbb::blocked_range<points_iterator> &r, ResultTuple res) -> ResultTuple
            {
                auto &[src, tgt] = res;
                src.reserve(r.size());
                tgt.reserve(r.size());
                for (const auto &point : r)
                {
                    Eigen::Vector3d closest_neighboors = GetClosestNeighboor(point);
                    if ((closest_neighboors - point).norm() < max_correspondance_distance)
                    {
                        src.emplace_back(point);
                        tgt.emplace_back(closest_neighboors);
                    }
                }
                return res;
            },
            // 2nd lambda: Parallel reduction
            [](ResultTuple a, const ResultTuple &b) -> ResultTuple
            {
                auto &[src, tgt] = a;
                const auto &[srcp, tgtp] = b;
                src.insert(src.end(), //
                           std::make_move_iterator(srcp.begin()), std::make_move_iterator(srcp.end()));
                tgt.insert(tgt.end(), //
                           std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
                return a;
            });

        return std::make_tuple(source, target);
    }

    std::vector<Eigen::Vector3d> CustomVoxelHashMap::Pointcloud() const
    {
        std::vector<Eigen::Vector3d> points;
        points.reserve(max_points_per_voxel_ * map_.size());
        for (const auto &[voxel, voxel_block] : map_)
        {
            (void)voxel;
            for (const auto &point : voxel_block.points)
            {
                points.push_back(point);
            }
        }
        return points;
    }

    void CustomVoxelHashMap::Update(const Vector3dVector &points, const Eigen::Vector3d &origin)
    {
        AddPoints(points);
        RemovePointsFarFromLocation(origin);
        if (++update_counter_ % clean_interval_ == 0)
        {
            // CleanMap(3);
            RemoveOutliers(0.04, 3);
            // TrimToMaxPoints(300000);
        }
    }

    void CustomVoxelHashMap::Update(const Vector3dVector &points, const Sophus::SE3d &pose)
    {
        Vector3dVector points_transformed(points.size());
        std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                       [&](const auto &point)
                       { return pose * point; });
        const Eigen::Vector3d &origin = pose.translation();
        Update(points_transformed, origin);
    }

    void CustomVoxelHashMap::AddPoints(const std::vector<Eigen::Vector3d> &points)
    {
        std::for_each(points.cbegin(), points.cend(), [&](const auto &point)
                      {
        auto voxel = Voxel((point / voxel_size_).template cast<int>());
        auto search = map_.find(voxel);
        if (search != map_.end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(point);
        } else {
            map_.insert({voxel, VoxelBlock{{point}, max_points_per_voxel_}});
        } });
    }

    void CustomVoxelHashMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin)
    {
        for (const auto &[voxel, voxel_block] : map_)
        {
            const auto &pt = voxel_block.points.front();
            const auto max_distance2 = max_distance_ * max_distance_;
            if ((pt - origin).squaredNorm() > (max_distance2))
            {
                map_.erase(voxel);
            }
        }
    }

    void CustomVoxelHashMap::RemoveOutliers(double search_radius, std::size_t min_neighbors)
    {
        const double radius_squared = search_radius * search_radius;

        // Iterate over the map and find outliers
        for (auto &[voxel, voxel_block] : map_)
        {
            auto &points = voxel_block.points;
            std::size_t i = 0;

            // Iterate over the points in the current voxel
            while (i < points.size())
            {
                const Eigen::Vector3d &point = points[i];
                int count = 0;

                const int kx = static_cast<int>(point[0] / voxel_size_);
                const int ky = static_cast<int>(point[1] / voxel_size_);
                const int kz = static_cast<int>(point[2] / voxel_size_);

                // Check neighboring voxels
                for (int dx = -1; dx <= 1; ++dx)
                {
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        for (int dz = -1; dz <= 1; ++dz)
                        {
                            Voxel neighbor_voxel{kx + dx, ky + dy, kz + dz};
                            auto search = map_.find(neighbor_voxel);
                            if (search == map_.end())
                                continue;

                            const auto &neighbor_block = search->second.points;
                            for (const auto &neighbor_point : neighbor_block)
                            {
                                if ((neighbor_point - point).squaredNorm() <= radius_squared)
                                {
                                    ++count;
                                    if (count >= min_neighbors)
                                        break;
                                }
                            }

                            if (count >= min_neighbors)
                                break;
                        }
                        if (count >= min_neighbors)
                            break;
                    }
                    if (count >= min_neighbors)
                        break;
                }

                // Remove the point if it doesn't have enough neighbors
                if (count < min_neighbors)
                {
                    auto search = map_.find(voxel);
                    if (search != map_.end())
                    {
                        auto &voxel_block = search.value();
                        voxel_block.RemovePointAt(i);
                    }
                }
                else
                {
                    ++i; // Only increment if the point is not removed
                }
            }

            // Remove the voxel if it is empty after point removal
            if (voxel_block.points.empty())
            {
                map_.erase(voxel); // Erase the voxel and move to the next
            }
        }
    }

    void CustomVoxelHashMap::CleanMap(std::size_t min_points_threshold)
    {
        for (auto it = map_.begin(); it != map_.end();)
        {
            const auto &voxel_block = it->second;
            if (voxel_block.points.size() < min_points_threshold)
            {
                it = map_.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

} // namespace kiss_icp