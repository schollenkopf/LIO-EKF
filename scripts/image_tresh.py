import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Set paths
input_folder = "../output/scenes_colmap/first6_calibrated_cleaned3/images/"  # Change to your folder containing images
output_folder = "../output/scenes_colmap/first6_calibrated_cleaned3/mask2/"  # Change to your desired output folder
os.makedirs(output_folder, exist_ok=True)

# Set global threshold value (adjust as needed)
THRESHOLD_VALUE = 20  # Change this based on your images

MORPH_KERNEL_SIZE = (50, 50)
GAUSSIAN_KERNEL_SIZE = (201, 201)
GAUSSIAN_SIGMA = 128


# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # # Read the image in grayscale
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # # Apply global threshold
        # _, binary_mask = cv2.threshold(image, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        # # Save the binary image
        # cv2.imwrite(output_path, binary_mask)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Convert to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Extract the L channel
        L_channel = lab_image[:, :, 0]

        brightness = L_channel.copy()

        kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
        brightness = (brightness / (255 / 15)).astype(np.uint8) * (255 / 15)
        # brightness = cv2.morphologyEx(brightness, cv2.MORPH_CLOSE, kernel)
        brightness = cv2.GaussianBlur(brightness, GAUSSIAN_KERNEL_SIZE, 0)
        mask = ((brightness.astype(np.int16)) * (-1) + 128) * 2

        L_channel = L_channel + mask
        lab_image[:, :, 0] = L_channel.clip(0, 255)
        cvt_img = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

        # Save the processed L channel as a grayscale image
        cv2.imwrite(output_path, brightness)
        cv2.imwrite(
            os.path.join(
                "../output/scenes_colmap/first6_calibrated_cleaned3/brightend/",
                filename,
            ),
            cvt_img,
        )

        print(f"Processed {filename}")

print("All images processed and saved in", output_folder)
