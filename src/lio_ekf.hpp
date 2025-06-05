// MIT License
//
// Copyright (c) 2024 Yibin Wu, Tiziano Guadagnino
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once
#include <Eigen/Dense>
#include <deque>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <vector>
#include <cmath>

#include "imuPropagation.hpp"
#include "kiss_icp/core/Threshold.hpp"
// #include "kiss_icp/core/VoxelHashMap.hpp"
#include "kiss_icp/pipeline/KissICP.hpp"
#include "lio_types.hpp"
#include "CustomVoxelHashMap.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/radius_outlier_removal.h>

// namespace Eigen
namespace lio_ekf
{

#define STATE_RANK 15
#define STATE_NOISE_RANK 12

  using Vector3dVector = std::vector<Eigen::Vector3d>;

  class LIOEKF
  {

  public:
    explicit LIOEKF(LIOPara &lio_para)
        : liopara_(lio_para), lio_map_(lio_para.voxel_size, lio_para.max_range,
                                       lio_para.max_points_per_voxel) {}

    LIOEKF()
        : lio_map_(liopara_.voxel_size, liopara_.max_range,
                   liopara_.max_points_per_voxel) {}

    ~LIOEKF() = default;

    void init();

    inline void addImuData(std::deque<lio_ekf::IMU> &imu_buffer_,
                           bool compensate = false)
    {
      const IMU &imu = imu_buffer_.front();
      imupre_ = imucur_;
      imucur_ = imu;
      // set current IMU timestamp as the current state timestamp
      imu_t_ = imucur_.timestamp;

      if (compensate)
      {
        imuCompensate(imucur_, imuerror_);
      }
      imu_buffer_.pop_front();
    }

    inline void
    addLidarData(std::deque<std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>> &lidar_buffer_,
                 std::deque<double> &lidar_time_buffer_,
                 std::deque<std_msgs::Header> &lidar_header_buffer_,
                 std::deque<std::vector<double>> &points_per_scan_time_buffer_,
                 float laser_up)
    {
      const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &point_pairs = lidar_buffer_.front();
      double timestamp = lidar_time_buffer_.front();

      lidar_header_ = lidar_header_buffer_.front();
      lidar_t_ = timestamp;
      curlaserup_ = laser_up;

      // Split the pair into two vectors
      curpoints_source_.clear();
      curpoints_map_.clear();
      curpoints_source_.reserve(point_pairs.size());
      curpoints_map_.reserve(point_pairs.size());

      for (const auto &pair : point_pairs)
      {
        curpoints_source_.push_back(pair.first);
        curpoints_map_.push_back(pair.second);
      }

      if (!points_per_scan_time_buffer_.empty())
      {
        timestamps_per_points_ = points_per_scan_time_buffer_.front();
        points_per_scan_time_buffer_.pop_front();
      }
      else
      {
        timestamps_per_points_ = std::vector<double>(curpoints_source_.size(), 0.0);
      }

      lidar_header_buffer_.pop_front();
      lidar_buffer_.pop_front();
      lidar_time_buffer_.pop_front();
    }

    inline void addLaserUpData(std::deque<double> &laser_up_buffer_,
                               std::deque<double> &laser_up_time_buffer_)
    {
      double timestamp = laser_up_time_buffer_.front();
      double range = laser_up_buffer_.front();
      laser_up_ = range;

      if (!initial_depth_set_)
      {

        initial_depth_ = range;
        last_distance_to_top_ = range;
        initial_depth_set_ = true;
      }

      laser_up_t_ = timestamp;
      laser_up_buffer_.pop_front();
      laser_up_time_buffer_.pop_front();
    }

    void newImuProcess();

    inline double getImutimestamp() const { return imu_t_; }

    inline double getLiDARtimestamp() const { return lidar_t_; }

    NavState getNavState();
    NavState getPredState();

    inline Eigen::MatrixXd getCovariance() { return Cov_; }

    inline std::vector<Eigen::Vector3d> LocalMap() const
    {
      return lio_map_.Pointcloud();
    };
    inline std::vector<Eigen::Vector3d> getFrame_w() const
    {
      return curpoints_w_;
    };
    inline std::vector<Eigen::Vector3d> getKetPoints_w() const
    {
      return keypoints_w_;
    };

    std_msgs::Header lidar_header_;

    bool lidar_updated_ = false; // if has done lidar update in the filter system

    inline bool lidarinOutage();

    Eigen::Matrix4d poseTran(const Eigen::Matrix4d pose1,
                             const Eigen::Matrix4d pose2);

    inline void TransformPoints(const Eigen::Matrix4d &T,
                                std::vector<Eigen::Vector3d> &points)
    {
      std::transform(points.cbegin(), points.cend(), points.begin(),
                     [&](const auto &point)
                     {
                       const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
                       const Eigen::Vector3d translation = T.block<3, 1>(0, 3);
                       return Eigen::Vector3d{R * point + translation};
                     });
    }

    Eigen::Vector3d deskewPoint(
        const Eigen::Vector3d &point, const double &timestamp,
        const std::vector<std::pair<double, Sophus::SE3d>> &posesWithinScan);

  private:
    void navStateInitialization(const NavState &initstate,
                                const NavState &initstate_std);

    int isToUpdate(double imutime1, double imutime2, double updatetime) const;

    void statePropagation(IMU &imupre, IMU &imucur);

    auto processScan();
    auto processLaserUp();
    void laserUpUpdate();
    void lidarUpdate();
    void lidarLadderUpdate();

    Vector3dVector findLadder(Vector3dVector downsampled_cloud);

    void ekfPredict(Eigen::Matrix15d &Phi, Eigen::Matrix15d &Qd);

    void ekfUpdate(Eigen::MatrixXd &dz, Eigen::MatrixXd &H, Eigen::MatrixXd &R);

    void stateFeedback();

    inline void checkStateCov()
    {

      for (int i = 0; i < STATE_RANK; i++)
      {
        if (Cov_(i, i) < 0)
        {
          std::cout << "Covariance is negative at " << std::setprecision(10)
                    << getImutimestamp() << " !" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
    }

    // void lio_initialization();
    Eigen::Matrix4d interp_pose(double t1, double t2, Eigen::Matrix4d pose1,
                                Eigen::Matrix4d pose2, double mid_t);

    // kiss-icp pipeline
    std::pair<Vector3dVector, Vector3dVector>
    Voxelize(const std::vector<Eigen::Vector3d> &frame) const;

    void initFirstLiDAR(const int lidarUpdateFlag);

    void resetCov(Eigen::Matrix15d &Cov);

  private:
    LIOPara liopara_;

    double imu_t_, last_update_t_, lidar_t_, first_lidar_t, laser_up_t_ = 0;

    // time align error threshold
    double TIME_ALIGN_ERR = 0.001;

    // last laser up measurement
    double laser_up_;
    double initial_depth_;
    double last_distance_to_top_;
    double laser_up_change_ = 0;
    bool initial_depth_set_ = false;

    // raw imudata
    IMU imupre_; // previous imu data
    IMU imucur_; // current imu data

    // imu state (position, velocity, attitude and imu error)
    BodyState bodystate_cur_;
    BodyState bodystate_pred_;
    BodyState bodystate_pre_;

    ImuError imuerror_;

    // ekf variables

    Eigen::Matrix15d Cov_;
    Eigen::Matrix12d State_Noise_Cov_;
    Eigen::Vector15d delta_x_;
    Eigen::Matrix15d Imu_Prediction_Covariance_ = Eigen::Matrix15d::Zero();

    // state ID and noise ID
    enum StateID
    {
      POS_ID = 0,
      VEL_ID = 3,
      ATT_ID = 6,
      GYRO_BIAS_ID = 9,
      ACC_BIAS_ID = 12
    };
    enum NoiseID
    {
      VEL_RANDOMWALK_ID = 0,
      ANGEL_RANDOMWALK_ID = 3,
      GYRO_BIAS_STD_ID = 6,
      ACC_BIAS_STD_ID = 9
    };

    kiss_icp::CustomVoxelHashMap lio_map_;

    Vector3dVector curpoints_, curpoints_w_, keypoints_w_, curpoints_source_, curpoints_map_;
    float curlaserup_;
    std::vector<double>
        timestamps_per_points_; // Timestamps for each points in one frame

    bool lio_initialized_ = false;
    bool is_first_imu_ = true;
    bool is_first_lidar_ = true;

    pcl::PointCloud<pcl::PointXYZ>::Ptr convertEigenToPCL(const std::vector<Eigen::Vector3d> &eigenCloud)
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud(new pcl::PointCloud<pcl::PointXYZ>);
      for (const auto &point : eigenCloud)
      {
        pclCloud->push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
      }
      pclCloud->is_dense = false;
      return pclCloud;
    }

    Vector3dVector convertPCLToEigen(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pclCloud)
    {
      Vector3dVector eigenCloud;
      for (const auto &point : pclCloud->points)
      {
        eigenCloud.emplace_back(point.x, point.y, point.z);
      }
      return eigenCloud;
    }

    bool isVerticalLine(const pcl::ModelCoefficients::Ptr &coefficients)
    {

      double dx = coefficients->values[3];
      double dy = coefficients->values[4];
      double dz = coefficients->values[5];

      double length = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (length == 0)
        return false; // Avoid division by zero

      dz /= length;

      double angle = std::acos(std::abs(dz)) * 180.0 / M_PI;

      return (dz > 0.9); // Allow small deviations from vertical
    }
  };
} // namespace lio_ekf