import os
import cv2
import mediapipe as mp
import dlib
import csv
import argparse

def process_frames(frames_folder):
    # Initialize MediaPipe and Dlib
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    detector = dlib.get_frontal_face_detector()

    # Derive output paths
    folder_name = os.path.basename(frames_folder.rstrip('/'))
    output_csv = f"{folder_name}_key_points.csv"
    marked_images_folder = f"{folder_name}_marked"

    # Ensure directories exist
    os.makedirs(marked_images_folder, exist_ok=True)

    # Prepare CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["frame_name"] + [f"x{i}" for i in range(468)] + [f"y{i}" for i in range(468)]
        writer.writerow(header)

        # Process each frame in the folder
        for frame_name in sorted(os.listdir(frames_folder)):
            frame_path = os.path.join(frames_folder, frame_name)
            if not frame_path.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            # Load image
            image = cv2.imread(frame_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces with Dlib
            faces = detector(gray_image)

            # If no faces detected, write a row with "No face detected"
            if len(faces) == 0:
                writer.writerow([frame_name] + ["No face detected"] * 468)
                continue

            # Process each detected face
            for face in faces:
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                face_roi = rgb_image[y1:y2, x1:x2]

                # Apply MediaPipe to the ROI
                results = face_mesh.process(face_roi)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        key_points = []
                        for landmark in face_landmarks.landmark:
                            x = int(landmark.x * (x2 - x1)) + x1
                            y = int(landmark.y * (y2 - y1)) + y1
                            key_points.extend([x, y])
                            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

                        # Write data to CSV
                        writer.writerow([frame_name] + key_points)

            # Save the marked image
            marked_image_path = os.path.join(marked_images_folder, frame_name)
            cv2.imwrite(marked_image_path, image)

    print(f"Face key points saved to {output_csv}")
    print(f"Marked images saved to {marked_images_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process frames for face key points detection.")
    parser.add_argument("frames_folder", type=str, help="Path to the folder containing frames.")
    args = parser.parse_args()

    if not os.path.isdir(args.frames_folder):
        print(f"Error: {args.frames_folder} is not a valid directory.")
        exit(1)

    process_frames(args.frames_folder)
