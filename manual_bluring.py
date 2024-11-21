import os
import cv2
import numpy as np
import pandas as pd
import argparse

# Global variables
drawing = False
mode = False  # False for circle, True for rectangle
eraser_mode = False  # Eraser mode
ix, iy = -1, -1
img = None
original_img = None  # Store the original image for erasing
shapes = []  # Store drawn shapes for blurring purposes
image_files = []  # List to store image filenames with full paths
log_data = []  # Store log data with frame_id and coordinates
current_image_index = 0  # Keep track of the current image being processed


# Function to handle the drawing event
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode, img, original_img, shapes, eraser_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if eraser_mode:  # Eraser mode
                img = original_img.copy()  # Reset to original image
                for shape in shapes:
                    if shape[0] == 'circle':  # Erase blur dot
                        cx, cy, radius = shape[1], shape[2], shape[3]
                        if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius:
                            shapes.remove(shape)  # Remove the shape
            else:
                if not mode:  # Apply blur for circles
                    radius = 15  # Radius of the blur dot
                    apply_blur_dot(x, y, radius)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if eraser_mode:
            return  # Don't store shapes when erasing, just modify the image
        if not mode:  # Apply blur for circles
            radius = 10  # Radius of the blur dot
            apply_blur_dot(x, y, radius)
            shapes.append(('circle', x, y, radius))  # Store blur dot for reference


# Function to apply a blur dot
def apply_blur_dot(x, y, radius):
    global img
    mask = np.zeros_like(img, dtype=np.uint8)  # Create a blank mask
    cv2.circle(mask, (x, y), radius, (255, 255, 255), -1)  # Draw a filled circle on the mask
    blurred_img = cv2.GaussianBlur(img, (31, 31), 0)  # Apply a Gaussian blur to the entire image
    img = np.where(mask > 0, blurred_img, img)  # Combine the blurred region with the original image


# Function to toggle between drawing and eraser modes
def toggle_eraser_mode():
    global eraser_mode
    eraser_mode = not eraser_mode
    if eraser_mode:
        print("Eraser mode activated. Click to erase areas.")
    else:
        print("Drawing mode activated. Press 'm' to toggle between circle/rectangle.")


# Function to process the CSV and show the problematic frames
def process_csv_for_face_detection(folder_path, modified_folder):
    global image_files
    try:
        # Find the first CSV file in the folder
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV file found in the folder: {folder_path}")
            return

        csv_file_path = os.path.join(folder_path, csv_files[0])
        df = pd.read_csv(csv_file_path)

        # Check if the 'face_detected' column exists
        if 'face_detected' not in df.columns:
            print(f"The file {csv_file_path} does not have a 'face_detected' column.")
            return

        # Filter rows where face_detected is -1
        problematic_frames = df[df['face_detected'] == -1]

        if problematic_frames.empty:
            print(f"No rows found with 'face_detected' as -1 in file: {csv_file_path}")
        else:
            print(f"\nFrames with 'face_detected' as -1 in file: {csv_file_path}")

            # Clear the image_files list
            image_files = []

            frames_folder = os.path.join(folder_path, "frames")

            # Find and append image paths corresponding to problematic frames
            for _, row in problematic_frames.iterrows():
                frame_id = int(row['frame_id'])  # Ensure frame_id is an integer
                image_name = f"frame_{frame_id:04d}.jpg"  # Format the filename
                image_path = os.path.join(frames_folder, image_name)

                if os.path.exists(image_path):
                    image_files.append((image_path, frame_id))
                else:
                    print(f"Warning: Image for frame_id {frame_id} not found at {image_path}")

            # Process the images
            if image_files:
                process_images(modified_folder)


    except Exception as e:
        print(f"Error processing CSV file: {e}")


# Function to process and show the image
def process_images(modified_folder):
    global img, original_img, shapes, current_image_index, log_data

    while current_image_index < len(image_files):
        image_path, frame_id = image_files[current_image_index]

        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found!")
            current_image_index += 1
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image {image_path}")
            current_image_index += 1
            continue

        original_img = img.copy()  # Store the original image for erasing

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_circle)

        while True:
            cv2.imshow('image', img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('m'):  # Toggle between rectangle and circle mode
                global mode
                mode = not mode
            elif key == ord('s'):  # Save the modified image
                save_image(image_path, modified_folder, frame_id)
                current_image_index += 1  # Move to the next image after saving
                break  # Break out of the inner while loop to process the next image
            elif key == ord('e'):  # Toggle between eraser and drawing modes
                toggle_eraser_mode()
            elif key == ord('c'):  # Clear drawn shapes (reset drawing)
                reset_drawing()
            elif key == 27:  # Escape key to exit
                break

        cv2.destroyAllWindows()

    # After processing all images, save the log and exit
    print("No more images to process.")
    save_log(modified_folder)


# Function to save the modified image and log data
def save_image(image_path, modified_folder, frame_id):
    global img, shapes, log_data

    # Save the modified image with its original name
    output_image_path = os.path.join(modified_folder, os.path.basename(image_path))
    cv2.imwrite(output_image_path, img)
    print(f"Modified image saved as {output_image_path}")

    # Log the shapes drawn on the image
    for shape in shapes:
        if shape[0] == 'circle':
            _, x, y, radius = shape
            log_data.append([frame_id, x, y, radius])


# Function to clear the painted regions
def reset_drawing():
    global img, original_img, shapes
    img = original_img.copy()  # Reload the original image
    shapes.clear()  # Clear the stored shapes


# Save the log data to a CSV file
def save_log(modified_folder):
    global log_data
    log_file_path = os.path.join(modified_folder, "log.csv")
    log_df = pd.DataFrame(log_data, columns=["frame_id", "x", "y", "radius"])
    log_df.to_csv(log_file_path, index=False)
    print(f"Log file saved as {log_file_path}")


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images based on CSV and blur problematic areas.")
    parser.add_argument("--folder", required=True, help="Path to the folder containing the CSV and frames folder.")
    parser.add_argument("--output_folder", required=True, help="Path to save modified images.")
    args = parser.parse_args()

    # Ensure output folder exists
    output_folder = os.path.join(args.folder, args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Process the CSV and images
    process_csv_for_face_detection(args.folder, output_folder)
