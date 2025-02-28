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
#include "CylindricalVoxelHashMap.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include <ros/ros.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/radius_outlier_removal.h>

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

CylindricalVoxelHashMap::Vector3dVectorTuple CylindricalVoxelHashMap::GetCorrespondences(
    const Vector3dVector &points, double max_correspondance_distance) const
{

    // Lambda Function to obtain the KNN of one point, maybe refactor

    auto GetClosestNeighboor = [&](const Eigen::Vector3d &point)
    {
        Eigen::Vector3d vec = point - axis_point_;
        float h = vec.dot(axis_dir_);
        Eigen::Vector3d projected_point = axis_point_ + h * axis_dir_;
        Eigen::Vector3d radial_vector = point - projected_point;
        float r = radial_vector.norm();
        float theta = std::atan2(radial_vector.y(), radial_vector.x());
        auto voxel = Eigen::Vector3i(std::round(r / radius_resolution_),
                                     std::round(theta / theta_resolution_),
                                     std::round(h / voxel_size_));
        // auto kx = static_cast<int>(point[0] / voxel_size_);
        // auto ky = static_cast<int>(point[1] / voxel_size_);
        // auto kz = static_cast<int>(point[2] / voxel_size_);
        auto kr = voxel[0];
        auto kt = voxel[1];
        auto kh = voxel[2];
        std::vector<CylindricalVoxel> voxels;
        voxels.reserve(18);

        for (int j = kt - 1; j < kt + 1 + 1; ++j)
        {
            for (int k = kh - 1; k < kh + 1 + 1; ++k)
            {
                voxels.emplace_back(kr, j, k);
            }
        }

        using Vector3dVector = std::vector<Eigen::Vector3d>;
        Vector3dVector neighboors;
        neighboors.reserve(18 * max_points_per_voxel_);
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

std::vector<Eigen::Vector3d> CylindricalVoxelHashMap::Pointcloud() const
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

void CylindricalVoxelHashMap::Update(const Vector3dVector &points, const Eigen::Vector3d &origin)
{
    if (!initialised_)
    {
        FitCylinder(points);
        initialised_ = true;
    }
    AddPoints(points);
    RemovePointsFarFromLocation(origin);
}

void CylindricalVoxelHashMap::Update(const Vector3dVector &points, const Sophus::SE3d &pose)
{
    Vector3dVector points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point)
                   { return pose * point; });
    const Eigen::Vector3d &origin = pose.translation();
    Update(points_transformed, origin);
}

void CylindricalVoxelHashMap::AddPoints(const std::vector<Eigen::Vector3d> &points)
{
    std::for_each(points.cbegin(), points.cend(), [&](const auto &point)
                  {
        Eigen::Vector3d vec = point - axis_point_;
        float h = vec.dot(axis_dir_);
        Eigen::Vector3d projected_point = axis_point_ + h * axis_dir_;
        Eigen::Vector3d radial_vector = point - projected_point;
        float r = radial_vector.norm();
        float theta = std::atan2(radial_vector.y(), radial_vector.x());
        auto voxel =  Eigen::Vector3i(std::round(r / radius_resolution_),
                            std::round(theta / theta_resolution_),
                            std::round(h / voxel_size_));
        auto search = map_.find(voxel);
        if (search != map_.end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(point);
        } else {
            map_.insert({voxel, CylindricalVoxelBlock{{point}, max_points_per_voxel_}});
        } });
}
void CylindricalVoxelHashMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin)
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

void CylindricalVoxelHashMap::FitCylinder(const std::vector<Eigen::Vector3d> &points)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &point : points)
    {
        cloud->push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
    }
    cloud->is_dense = false;
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

    // Estimate point normals
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud);
    ne.setKSearch(20);
    ne.compute(*cloud_normals);

    // Find cylinder
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setNormalDistanceWeight(0.1);
    seg.setMaxIterations(500);
    seg.setDistanceThreshold(0.13);
    // seg.setDistanceThreshold(0.23);
    seg.setRadiusLimits(2, 8);
    seg.setInputCloud(cloud);
    seg.setInputNormals(cloud_normals);
    seg.segment(*inliers, *coefficients);

    axis_point_ = Eigen::Vector3d(coefficients->values[0],
                                  coefficients->values[1],
                                  coefficients->values[2]);
    ROS_WARN_STREAM("FOUND AXIS POINT: " << axis_point_);
    axis_dir_ = Eigen::Vector3d(coefficients->values[3],
                                coefficients->values[4],
                                coefficients->values[5]);
    ROS_WARN_STREAM("FOUND AXIS Dir: " << axis_dir_);
    axis_dir_ = axis_dir_.normalized();
}