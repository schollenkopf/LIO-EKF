#!/bin/bash
# Usage Example:
#docker build . --file turbine_laser_slam/docker/Dockerfile  -t ros_melodic_slam
#http://localhost:8080/vnc.html
docker run -it --rm --net=ros --env="DISPLAY=novnc:0.0" --env="ROS_MASTER_URI=http://roscore:11311" --volume="${PWD}/LIO-EKF":"/catkin_ws/src/LIO-EKF":rw  thesis_noetic_base 

#bash -c "cd catkin_ws && catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release" 