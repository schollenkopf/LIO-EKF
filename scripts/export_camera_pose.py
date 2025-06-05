#!/usr/bin/env python3

import rospy
import tf
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from rosgraph_msgs.msg import Clock  # Import the correct message type

# File path to world -> base_link poses
base_link_poses_file = (
    "../output/ouster/esbjerg_preprocessed/odo_tum.txt"  # seems to output in frame front,left,up
)

# Load world -> base_link poses
df_base_link = pd.read_csv(
    base_link_poses_file,
    delim_whitespace=True,
    comment="#",
    names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"],
)

# Prepare storage for world -> camera poses
data = []


def clock_callback(msg):
    global current_time
    current_time = msg.clock


def compute_camera_pose():
    rospy.init_node("camera_pose_extractor")

    tf_listener = tf.TransformListener()
    rate = rospy.Rate(50)  # 50 Hz for better timestamp matching

    # Wait for the /clock topic to become available and subscribe to it
    rospy.Subscriber("/clock", Clock, clock_callback)

    # Wait for the first clock message
    rospy.wait_for_message("/clock", Clock)

    # Ensure the use of simulated time
    rospy.set_param("/use_sim_time", True)

    for index, row in df_base_link.iterrows():
        timestamp = rospy.Time.from_sec(row["timestamp"])

        # Wait until the clock time is after the desired timestamp
        while current_time < timestamp:
            rospy.loginfo(f"Waiting for clock to reach {timestamp.to_sec()}...")
            rate.sleep()

        try:
            # tf_listener.waitForTransform(
            #     "base_link", "camera", timestamp, rospy.Duration(0.01)
            # )
            trans, rot = tf_listener.lookupTransform(
                "camera_optical", "base_link", timestamp
            )

            # Convert world -> base_link pose
            world_T_base = np.eye(4)
            world_R_base = R.from_quat(
                [row["qx"], row["qy"], row["qz"], row["qw"]]
            ).as_matrix()
            world_T_base[:3, 3] = [row["x"], row["y"], row["z"]]
            world_T_base[:3, :3] = world_R_base

            # Convert base_link -> camera transform
            base_T_cam = np.eye(4)
            base_T_cam[:3, 3] = trans
            base_T_cam[:3, :3] = R.from_quat(rot).as_matrix()

            # Compute world -> camera pose
            world_T_cam = base_T_cam @ world_T_base
            # cam_translation = world_T_cam[:3, 3]
            cam_rotation = R.from_matrix(world_T_cam[:3, :3]).as_quat()
            world_R_cam = R.from_matrix(world_T_cam[:3, :3]).as_matrix()
            cam_position_world = -world_R_cam @ world_T_cam[:3, 3]
            # Store result
            data.append(
                [
                    row["timestamp"],
                    cam_position_world[0],  # Corrected x in world frame
                    cam_position_world[1],  # Corrected y in world frame
                    cam_position_world[2],  # Corrected z in world frame
                    cam_rotation[0],  # qx in world frame
                    cam_rotation[1],  # qy in world frame
                    cam_rotation[2],  # qz in world frame
                    cam_rotation[3],  # qw in world frame
                    row["x"],
                    row["y"],
                    row["z"],
                    row["qx"],
                    row["qy"],
                    row["qz"],
                    row["qw"],
                    trans[0],
                    trans[1],
                    trans[2],
                    rot[0],
                    rot[1],
                    rot[2],
                    rot[3],
                ]
            )

            rospy.loginfo(f"Processed timestamp: {row['timestamp']}")

        except:
            rospy.logwarn(f"Transform not available for timestamp {row['timestamp']}")

        rate.sleep()

    # Save to CSV
    df_camera = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw",
            "bx",
            "by",
            "bz",
            "bqx",
            "bqy",
            "bqz",
            "bqw",
            "cx",
            "cy",
            "cz",
            "cqx",
            "cqy",
            "cqz",
            "cqw",
        ],
    )
    csv_filename = "world_camera_poses_esbjerg_preprocessed.csv"
    df_camera.to_csv(csv_filename, index=False)
    rospy.loginfo(f"Camera poses saved to {csv_filename}")


if __name__ == "__main__":
    # Initialize the current_time variable to ensure it's defined globally
    current_time = rospy.Time(0)

    compute_camera_pose()

# apt install python3-pip
# pip3 install pandas scipy
# src/LIO_EKF/scripts:
# rosrun lio_ekf export_camera_pose.py
