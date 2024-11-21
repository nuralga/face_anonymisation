import argparse
import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd


def process_video(video_path, output_path, save_frames=False, save_avi=False, save_mp4=False, detection_confidence=0.5,
                  tracking_confidence=0.5, scale_factor=1.5):
    # Prepare output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Handle frames-specific subfolder
    frames_dir = os.path.join(output_path, "frames") if save_frames else None
    if save_frames and not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Initialize MediaPipe Face Detection and FaceMesh
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=detection_confidence)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=detection_confidence,
                                      min_tracking_confidence=tracking_confidence)

    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare video writers if needed
    avi_writer = None
    mp4_writer = None
    if save_avi:
        avi_writer = cv2.VideoWriter(
            os.path.join(output_path, "output.avi"),
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (width, height)
        )
    if save_mp4:
        mp4_writer = cv2.VideoWriter(
            os.path.join(output_path, "output.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

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

        # Save the frame
        if save_frames:
            frame_path = os.path.join(frames_dir, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(frame_path, image)
        if avi_writer:
            avi_writer.write(image)
        if mp4_writer:
            mp4_writer.write(image)

    # Save log data to CSV
    log_df = pd.DataFrame(log_data, columns=["frame_id", "x", "y", "height", "width", "face_detected"])
    log_csv_path = os.path.join(output_path, "log.csv")
    log_df.to_csv(log_csv_path, index=False)

    # Cleanup
    cap.release()
    face_detection.close()
    face_mesh.close()
    if avi_writer:
        avi_writer.release()
    if mp4_writer:
        mp4_writer.release()

    print(f"Processing completed. Logs saved in: {output_path}")
    if save_frames:
        print(f"Frames saved in: {frames_dir}")
    if avi_writer:
        print(f"Video saved as: output.avi")
    if mp4_writer:
        print(f"Video saved as: output.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video to detect and blur faces.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_path", type=str, help="Path to save the results.")
    parser.add_argument("--save_frames", action="store_true", help="Save processed frames as individual images.")
    parser.add_argument("--save_avi", action="store_true", help="Save the output video in AVI format.")
    parser.add_argument("--save_mp4", action="store_true", help="Save the output video in MP4 format.")
    parser.add_argument("--detection_confidence", type=float, default=0.5,
                        help="Minimum confidence for face detection.")
    parser.add_argument("--tracking_confidence", type=float, default=0.5,
                        help="Minimum confidence for landmark tracking.")
    parser.add_argument("--scale_factor", type=float, default=1.5, help="Scale factor for bounding box expansion.")

    args = parser.parse_args()

    process_video(
        args.video_path,
        args.output_path,
        save_frames=args.save_frames,
        save_avi=args.save_avi,
        save_mp4=args.save_mp4,
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence,
        scale_factor=args.scale_factor
    )
# python task_01_06.py "resistive_band_010.mp4" "resistive_band_010" --save_frames --save_avi --save_mp4