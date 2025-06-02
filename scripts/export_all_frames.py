import cv2
import os

# OG flight:
# video_path = "../../data/a107ac6c-305a-4208-a5a4-b4ef41c68c60/video/standalone_session_a107ac6c-305a-4208-a5a4-b4ef41c68c60.mp4"

# Esbjerg
video_path = "../../data/flight-esbjerg/standalone_session_1889b63b-12f0-4678-afe9-1fdc43675ce7.mp4"


out_folder = "scenes_monoGS/esbjerg"
output_image_dir = "../output/" + out_folder + "/images"

os.makedirs(output_image_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)

frame_idx = 0
target_width = int(1920 / 2)
target_height = int(1080 / 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (target_width, target_height))
    filename = os.path.join(output_image_dir, f"frame_{frame_idx:05d}.jpg")
    cv2.imwrite(filename, resized_frame)

    frame_idx += 1

cap.release()
print(f"Saved {frame_idx} frames to '{output_image_dir}'")
