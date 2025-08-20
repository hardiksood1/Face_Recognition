import cv2
from ultralytics import YOLO
from face_utils import recognize_face
import os
import torch

def process_video(input_path, output_dir, model, known_face_encodings, known_face_names,
                  resize_scale=0.5, frame_skip=1):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("❌ Unable to open video source")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))
    out_width, out_height = int(width*resize_scale), int(height*resize_scale)

    output_path = os.path.join(output_dir, f"processed_{os.path.basename(input_path)}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Resize for faster detection
        small_frame = cv2.resize(frame, (out_width, out_height))

        # YOLO inference
        results = model(small_frame, verbose=False)

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Scale coordinates back to original frame
                x1, y1, x2, y2 = [int(v/resize_scale) for v in [x1, y1, x2, y2]]
                face_crop = frame[y1:y2, x1:x2]

                name = recognize_face(face_crop, known_face_encodings, known_face_names) if face_crop.size > 0 else "Unknown"

                # Draw box + label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({conf:.2f})", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    return output_path