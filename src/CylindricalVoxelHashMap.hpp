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
//
// NOTE: This implementation is heavily inspired in the original CT-ICP VoxelHashMap implementation,
// although it was heavily modifed and drastically simplified, but if you are using this module you
// should at least acknoowledge the work from CT-ICP by giving a star on GitHub
#pragma once

#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

struct CylindricalVoxelHashMap
{
    using Vector3dVector = std::vector<Eigen::Vector3d>;
    using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;
    using CylindricalVoxel = Eigen::Vector3i;
    struct CylindricalVoxelBlock
    {
        // buffer of points with a max limit of n_points
        std::vector<Eigen::Vector3d> points;
        int num_points_;
        inline void AddPoint(const Eigen::Vector3d &point)
        {
            if (points.size() < static_cast<size_t>(num_points_))
                points.push_back(point);
        }
    };
    struct CylindricalVoxelHash
    {
        size_t operator()(const CylindricalVoxel &voxel) const
        {
            const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
            return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
        }
    };

    explicit CylindricalVoxelHashMap(double voxel_size, double max_distance, int max_points_per_voxel)
        : voxel_size_(voxel_size),
          max_distance_(max_distance),
          max_points_per_voxel_(max_points_per_voxel) {}

    Vector3dVectorTuple GetCorrespondences(const Vector3dVector &points,
                                           double max_correspondance_distance) const;
    inline void Clear() { map_.clear(); }
    inline bool Empty() const { return map_.empty(); }
    void Update(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &origin);
    void Update(const std::vector<Eigen::Vector3d> &points, const Sophus::SE3d &pose);
    void AddPoints(const std::vector<Eigen::Vector3d> &points);
    void FitCylinder(const std::vector<Eigen::Vector3d> &points);
    // CylindricalVoxel CartesianToCylindrical(const Eigen::Vector3f &point, Eigen::Vector3f axis_point_
    //                                                                           Eigen::Vector3f axis_dir_);
    void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
    std::vector<Eigen::Vector3d> Pointcloud() const;

    double theta_resolution_ = 0.3;
    double radius_resolution_ = 0.05;
    double voxel_size_;
    double max_distance_;
    int max_points_per_voxel_;
    bool initialised_ = false;
    Eigen::Vector3d axis_point_;
    Eigen::Vector3d axis_dir_;
    tsl::robin_map<CylindricalVoxel, CylindricalVoxelBlock, CylindricalVoxelHash> map_;
};

// Eigen::Vector3i CartesianToCylindrical(const Eigen::Vector3d &point,
//                                        const Eigen::Vector3d &axis_point_,
//                                        const Eigen::Vector3d &axis_dir_,
//                                        double voxel_size_)
// {
//     // Ensure axis_dir is normalized
//     Eigen::Vector3d axis_unit = axis_dir_.normalized();

//     // Vector from the axis point to the input point
//     Eigen::Vector3d vec = point - axis_point_;

//     // Project vec onto the axis to find height (h)
//     float h = vec.dot(axis_unit);

//     // Compute the projected point on the cylinder axis
//     Eigen::Vector3d projected_point = axis_point_ + h * axis_unit;

//     // Compute the radial vector (vector perpendicular to the axis)
//     Eigen::Vector3d radial_vector = point - projected_point;

//     // Radius (r) is the length of the radial vector
//     float r = radial_vector.norm();

//     // Compute angle theta in the XY plane relative to the cylinder axis
//     float theta = std::atan2(radial_vector.y(), radial_vector.x());

//     // Scale and cast to integer for voxel indexing
//     return Eigen::Vector3i(std::round(r / voxel_size_),
//                            std::round(theta / voxel_size_),
//                            std::round(h / voxel_size_));
// }
