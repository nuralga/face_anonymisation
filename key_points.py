import os
import cv2
import mediapipe as mp
import dlib
import csv

# Initialize MediaPipe and Dlib
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
detector = dlib.get_frontal_face_detector()

# Folder containing frames
frames_folder = "/Users/nurzh/AGH/lip-reading/new_kuchiyomi/word_frame"  # Replace with the path to your folder
output_csv = "face_key_points/output.csv"  # Replace with the desired output CSV file path
marked_images_folder = "marked_frames"    # Folder to save the marked images

# Ensure the directories for the CSV file and marked images exist
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
os.makedirs(marked_images_folder, exist_ok=True)

# Prepare CSV file
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header: frame_name and 468 x,y points for MediaPipe
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
            continue  # Skip to next frame

        # Process each detected face
        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            face_roi = rgb_image[y1:y2, x1:x2]

            # Apply MediaPipe to the ROI
            results = face_mesh.process(face_roi)

            # Extract key points and mark them on the image
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    key_points = []
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * (x2 - x1)) + x1
                        y = int(landmark.y * (y2 - y1)) + y1
                        key_points.extend([x, y])

                        # Draw the landmark points on the image
                        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Green points

                    # Write data to CSV
                    writer.writerow([frame_name] + key_points)

        # Save the marked image in the marked_images folder
        marked_image_path = os.path.join(marked_images_folder, frame_name)
        cv2.imwrite(marked_image_path, image)

print(f"Face key points saved to {output_csv}")
print(f"Marked images saved to {marked_images_folder}")
