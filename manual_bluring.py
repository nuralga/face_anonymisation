import os
import cv2
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog

# Global variables
drawing = False
mode = False  # False for circle, True for rectangle
eraser_mode = False  # Eraser mode
ix, iy = -1, -1
img = None
original_img = None  # Store the original image for erasing
shapes = []  # Store drawn shapes for blurring purposes
current_image_index = 0  # Index for cycling through images
image_files = []  # List to store image filenames with full paths


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
                    # Erase only if the eraser is touching the shape
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


# Apply blur to the drawn shapes
def apply_blurring(image, shape_coords, blur_radius=15):
    for shape in shape_coords:
        if shape[0] == 'rect':  # Rectangle
            x1, y1, x2, y2 = shape[1], shape[2], shape[3], shape[4]
            roi = image[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (blur_radius, blur_radius), 0)
            image[y1:y2, x1:x2] = blurred_roi
        elif shape[0] == 'circle':  # Circle
            x, y, _, _ = shape[1], shape[2], shape[3], shape[4]
            cv2.circle(image, (x, y), 15, (0, 0, 0), -1)  # Manual blur effect
    return image


# Function to process the CSV and show the problematic frames
def process_csv_for_face_detection(file_path, frames_folder):
    global image_files
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if the 'face_detected' column exists
        if 'face_detected' not in df.columns:
            print(f"The file {file_path} does not have a 'face_detected' column.")
            return

        # Filter rows where face_detected is -1
        problematic_frames = df[df['face_detected'] == -1]

        if problematic_frames.empty:
            print(f"No rows found with 'face_detected' as -1 in file: {file_path}")
        else:
            print(f"\nFrames with 'face_detected' as -1 in file: {file_path}")

            # Clear the image_files list
            image_files = []

            # Find and append image paths corresponding to problematic frames
            for _, row in problematic_frames.iterrows():
                frame_id = int(row['frame_id'])  # Ensure frame_id is an integer
                image_name = f"frame_{frame_id:04d}.jpg"  # Format the filename
                image_path = os.path.join(frames_folder, image_name)

                if os.path.exists(image_path):
                    image_files.append(image_path)
                else:
                    print(f"Warning: Image for frame_id {frame_id} not found at {image_path}")

            # Display the problematic frames and associated image paths
            print("Images to be processed:")
            for image_file in image_files:
                print(image_file)

            # Process the first problematic image
            if image_files:
                process_image(image_files[0])

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def find_image_file(frame_id, frames_folder):
    # Find the image file corresponding to the frame_id in the "frames" folder
    image_file = next((img for img in os.listdir(frames_folder) if f"frame_{frame_id:04d}" in img), None)
    return image_file


# Function to process and show the image
def process_image(image_path):
    global img, original_img, shapes, current_image_index

    # Check if the image path exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return

    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Failed to load image {image_path}")
        return

    original_img = img.copy()  # Store the original image for erasing

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while True:
        # Display the image with the drawn shapes
        cv2.imshow('image', img)

        # Wait for user input for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('m'):  # Toggle between rectangle and circle mode
            global mode
            mode = not mode
        elif key == ord('s'):  # Save the modified image
            save_image()
            current_image_index += 1  # Move to the next image after saving
            if current_image_index >= len(image_files):
                print("No more images to process.")
                break  # Exit if there are no more images
            else:
                img = cv2.imread(image_files[current_image_index])  # Load the next image
                original_img = img.copy()
        elif key == ord('e'):  # Toggle between eraser and drawing modes
            toggle_eraser_mode()
        elif key == ord('c'):  # Clear drawn shapes (i.e., reset drawing)
            reset_drawing()
        elif key == 27:  # Escape key to exit
            break

    cv2.destroyAllWindows()


# Function to save the modified image
def save_image():
    global img, image_files, current_image_index

    # Get the current image path
    current_image_path = image_files[current_image_index]

    # Generate the modified image path
    modified_image_path = current_image_path.replace('.jpg', '_modified.jpg').replace('.png', '_modified.png')

    # Save the modified image
    cv2.imwrite(modified_image_path, img)
    print(f"Modified image saved as {modified_image_path}")



# Function to clear the painted regions
def reset_drawing():
    global img, original_img, shapes
    img = original_img.copy()  # Reload the original image
    shapes.clear()  # Clear the stored shapes


# Open a folder selection dialog to select folder containing CSV and frames
def select_folder():
    root = Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(title="Select Folder Containing CSV Files")
    if folder_path:
        print(f"Selected folder: {folder_path}")
        find_and_process_csv_files(folder_path)
    else:
        print("No folder selected.")


# Function to find and process all CSV files in the folder
def find_and_process_csv_files(folder_path):
    global image_files
    files = os.listdir(folder_path)
    csv_files = [file for file in files if file.endswith('.csv')]
    frames_folder = os.path.join(folder_path, "frames")

    # Create a list of image files in the frames folder corresponding to problematic frames
    if not csv_files:
        print("No CSV files found in the folder.")
        return

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        process_csv_for_face_detection(file_path, frames_folder)


# Run the folder selection
if __name__ == "__main__":
    select_folder()
