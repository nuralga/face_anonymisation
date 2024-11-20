import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def process_video(video_path):
    # Extract video name and prepare output directory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = video_name
    frames_dir = os.path.join(output_dir, "frames")  # Create a 'frames' folder inside the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)  # Create 'frames' directory if it doesn't exist

    # Initialize MediaPipe Face Detection and FaceMesh
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare log file
    log_data = []

    frame_id = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        frame_id += 1
        img_h, img_w, _ = image.shape

        # Convert the color space from BGR to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = face_detection.process(image_rgb)

        face_detected = 0
        if detections.detections:
            for detection in detections.detections:
                # Get the bounding box coordinates for the face
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * img_w)
                y_min = int(bboxC.ymin * img_h)
                box_width = int(bboxC.width * img_w)
                box_height = int(bboxC.height * img_h)

                # Expand the bounding box by a scale factor for better landmark detection
                scale_factor = 1.5
                x_center, y_center = x_min + box_width // 2, y_min + box_height // 2
                scaled_width, scaled_height = int(box_width * scale_factor), int(box_height * scale_factor)
                x_start, y_start = max(0, x_center - scaled_width // 2), max(0, y_center - scaled_height // 2)
                x_end, y_end = min(img_w, x_center + scaled_width // 2), min(img_h, y_center + scaled_height // 2)

                # Crop the face area and process with FaceMesh
                face_crop = image_rgb[y_start:y_end, x_start:x_end]
                face_results = face_mesh.process(face_crop)

                # Convert back to BGR for OpenCV operations
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                # Mask for blurring
                mask = np.zeros((img_h, img_w), dtype=np.uint8)

                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        # Collect scaled-up landmark points
                        face_points = [(int(lm.x * scaled_width + x_start), int(lm.y * scaled_height + y_start))
                                       for lm in face_landmarks.landmark]

                        # Define face region using convex hull and expand
                        hull = cv2.convexHull(np.array(face_points))
                        cv2.fillPoly(mask, [hull], 255)

                        # Apply strong blur
                        blurred_image = cv2.GaussianBlur(image, (101, 101), 0)
                        image = np.where(mask[:, :, None] == 255, blurred_image, image)

                # Log the face details
                log_data.append([frame_id, x_min, y_min, box_height, box_width, 1])
                face_detected = 1

        if not face_detected:
            log_data.append([frame_id, -1, -1, -1, -1, -1])

        # Save the frame to the 'frames' directory
        frame_path = os.path.join(frames_dir, f"frame_{frame_id:04d}.jpg")
        cv2.imwrite(frame_path, image)

    # Save log data to CSV
    log_df = pd.DataFrame(log_data, columns=["frame_id", "x", "y", "height", "width", "face_detected"])
    log_csv_path = os.path.join(output_dir, f"{video_name}_log.csv")
    log_df.to_csv(log_csv_path, index=False)

    # Cleanup
    cap.release()
    face_detection.close()
    face_mesh.close()
    print(f"Processing completed. Frames and log saved in: {output_dir}")


if __name__ == "__main__":
    # Create a file dialog to select a video file
    Tk().withdraw()  # Hide the root window
    video_path = askopenfilename(
        title="Select a Video File",
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )

    if video_path:
        process_video(video_path)
    else:
        print("No file selected.")
