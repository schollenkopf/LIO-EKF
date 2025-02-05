docker run  -it --rm --name=ouster  --net=ros --env="DISPLAY=novnc:0.0" --env="ROS_MASTER_URI=http://roscore:11311" thesis_ouster
#roslaunch ouster_ros replay.launch