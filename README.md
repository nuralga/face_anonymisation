# Face Anonymisation for Video

Tools to blur faces in video for privacy-safe datasets. Blurring is automatic, using MediaPipe face detection and a landmark-based face mask. Frames where detection fails can be fixed by hand in a small review tool. A separate script extracts facial key points to CSV.

## How it works

1. **Automatic blurring** (`blur_faces.py`): MediaPipe detects the face, the bounding box is expanded, and Face Mesh landmarks inside it give a convex-hull mask of the face. A strong Gaussian blur is applied inside the mask only. Each frame is logged to `log.csv`, with `face_detected = -1` when no face was found.
2. **Manual review** (`manual_bluring.py`): opens only the frames that `log.csv` marks as missed. You paint blur with the mouse, and the tool saves the fixed frames plus a log of every blurred region.
3. **Key-point extraction** (`key_points.py`, `key_points_from_video.py`): a dlib face detector feeds MediaPipe Face Mesh, which exports 468 landmarks per frame to CSV and saves annotated frames. The video version also uses MediaPipe Pose to skip frames with no person.

## Usage

```bash
pip install opencv-python mediapipe dlib numpy pandas

# 1. Blur faces automatically; save frames + MP4
python blur_faces.py input.mp4 out_dir --save_frames --save_mp4

# 2. Fix frames where no face was detected
python manual_bluring.py --folder out_dir --output_folder fixed
#    mouse: paint blur | e: eraser | c: clear | s: save & next | Esc: quit

# 3. Export facial landmarks to CSV
python key_points_from_video.py input.mp4
python key_points.py frames_folder/
```

Optional flags for `main.py`: `--save_avi`, `--detection_confidence`, `--tracking_confidence`, `--scale_factor` (bounding-box expansion, default 1.5).

## Stack

Python · OpenCV · MediaPipe (Face Detection, Face Mesh, Pose) · dlib · pandas
