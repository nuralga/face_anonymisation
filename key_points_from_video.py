import cv2
import os
import argparse
import mediapipe as mp
import csv
import dlib

def process_video(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_file}.")
        exit(1)

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_csv = f"{video_name}.csv"
    marked_images_folder = f"{video_name}_marked"
    no_pose_folder = f"{video_name}_no_pose"
    no_face_folder = f"{video_name}_no_face"
    os.makedirs(marked_images_folder, exist_ok=True)
    os.makedirs(no_pose_folder, exist_ok=True)
    os.makedirs(no_face_folder, exist_ok=True)

    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    pose = mp_pose.Pose(static_image_mode=True)
    cascade_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    dlib_detector = dlib.get_frontal_face_detector()

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["frame_name"] + [f"x{i}" for i in range(468)] + [f"y{i}" for i in range(468)]
        writer.writerow(header)

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Error: Failed to read frame {frame_number}.")
                break

            frame_name = f"frame_{frame_number:04d}.jpg"
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Check for human pose
            results_pose = pose.process(rgb_frame)
            if not results_pose.pose_landmarks:
                print(f"No pose detected in frame {frame_number}. Skipping frame.")
                no_pose_path = os.path.join(no_pose_folder, frame_name)
                cv2.imwrite(no_pose_path, frame)
                frame_number += 1
                continue

            # Check for face presence using cascade detector
            faces = cascade_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # If no faces detected by cascade, use dlib
            if len(faces) == 0:
                dlib_faces = dlib_detector(gray_frame)
                faces = [(face.left(), face.top(), face.width(), face.height()) for face in dlib_faces]

            # If pose exists but no face detected
            if len(faces) == 0:
                print(f"Human pose detected but no face in frame {frame_number}.")
                writer.writerow([frame_name] + [-1] * 936)  # -1 for all keypoints
                no_face_path = os.path.join(no_face_folder, frame_name)
                cv2.imwrite(no_face_path, frame)
                frame_number += 1
                continue

            for (x1, y1, w, h) in faces:
                x2, y2 = x1 + w, y1 + h
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    print(f"Error: Invalid face bounding box in frame {frame_number}.")
                    continue

                face_roi = rgb_frame[y1:y2, x1:x2]
                results_face = face_mesh.process(face_roi)

                if results_face.multi_face_landmarks:
                    for face_landmarks in results_face.multi_face_landmarks:
                        key_points = []
                        for landmark in face_landmarks.landmark:
                            x = int(landmark.x * (x2 - x1)) + x1
                            y = int(landmark.y * (y2 - y1)) + y1
                            key_points.extend([x, y])
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                        writer.writerow([frame_name] + key_points)

            marked_image_path = os.path.join(marked_images_folder, frame_name)
            cv2.imwrite(marked_image_path, frame)
            frame_number += 1

    cap.release()
    print(f"Face key points saved to {output_csv}")
    print(f"Marked images saved to {marked_images_folder}")
    print(f"Frames with no pose saved to {no_pose_folder}")
    print(f"Frames with no faces but poses detected saved to {no_face_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process video for face keypoints and pose detection.")
    parser.add_argument("video_file", type=str, help="Path to the input video file.")
    args = parser.parse_args()
    process_video(args.video_file)
