import os
import cv2
import mediapipe as mp
import dlib
import csv
import argparse

def process_video(video_file):
    # Initialize MediaPipe Pose and Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    pose = mp_pose.Pose(static_image_mode=True)
    detector = dlib.get_frontal_face_detector()

    # Derive output paths
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_csv = f"{video_name}_key_points.csv"
    marked_images_folder = f"{video_name}_marked"
    no_pose_folder = f"{video_name}_no_pose"
    no_face_folder = f"{video_name}_no_face"

    # Ensure directories exist
    os.makedirs(marked_images_folder, exist_ok=True)
    os.makedirs(no_pose_folder, exist_ok=True)
    os.makedirs(no_face_folder, exist_ok=True)

    # Prepare CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["frame_name"] + [f"x{i}" for i in range(468)] + [f"y{i}" for i in range(468)]
        writer.writerow(header)

        # Open video file
        cap = cv2.VideoCapture(video_file)
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_name = f"frame_{frame_number:04d}.jpg"

            # Convert frame to appropriate formats
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect human pose
            results_pose = pose.process(rgb_frame)
            if not results_pose.pose_landmarks:
                print(f"No pose detected in frame {frame_number}. Skipping frame.")
                no_pose_path = os.path.join(no_pose_folder, frame_name)
                cv2.imwrite(no_pose_path, frame)
                frame_number += 1
                continue

            # Detect faces with Dlib
            faces = detector(gray_frame)

            # If no faces detected, write -1 for all key points and save to no_face_folder
            if len(faces) == 0:
                print(f"Pose detected but no face in frame {frame_number}.")
                writer.writerow([frame_name] + [-1] * 936)
                no_face_path = os.path.join(no_face_folder, frame_name)
                cv2.imwrite(no_face_path, frame)
                frame_number += 1
                continue

            # Select the largest detected face (only one person expected)
            largest_face = max(faces, key=lambda face: (face.right() - face.left()) * (face.bottom() - face.top()))

            # Get face coordinates
            x1, y1, x2, y2 = largest_face.left(), largest_face.top(), largest_face.right(), largest_face.bottom()

            # Validate face bounding box
            if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                print(f"Invalid face bounding box in frame {frame_number}: ({x1}, {y1}, {x2}, {y2}). Skipping.")
                frame_number += 1
                continue

            face_roi = rgb_frame[y1:y2, x1:x2]

            # Apply MediaPipe to the ROI
            results = face_mesh.process(face_roi)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    key_points = []
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * (x2 - x1)) + x1
                        y = int(landmark.y * (y2 - y1)) + y1
                        key_points.extend([x, y])
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                    # Write data to CSV
                    writer.writerow([frame_name] + key_points)

            # Save the marked image
            marked_image_path = os.path.join(marked_images_folder, frame_name)
            cv2.imwrite(marked_image_path, frame)
            frame_number += 1

    cap.release()
    print(f"Face key points saved to {output_csv}")
    print(f"Marked images saved to {marked_images_folder}")
    print(f"Frames with no pose saved to {no_pose_folder}")
    print(f"Frames with no faces but poses detected saved to {no_face_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for face key points and pose detection.")
    parser.add_argument("video_file", type=str, help="Path to the input video file.")
    args = parser.parse_args()

    if not os.path.isfile(args.video_file):
        print(f"Error: {args.video_file} is not a valid file.")
        exit(1)

    process_video(args.video_file)
