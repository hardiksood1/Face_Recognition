import os
import torch
import gradio as gr
from ultralytics import YOLO
from config import FACE_MODEL_PATH, KNOWN_PEOPLE_DIR, OUTPUT_DIR
from face_utils import load_known_faces
from video_processor import process_video

# ───────────── DEVICE SETUP ─────────────
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ───────────── LOAD MODELS ─────────────
model = YOLO(FACE_MODEL_PATH)
model.to(device)

# Load known faces
known_face_encodings, known_face_names = load_known_faces(KNOWN_PEOPLE_DIR)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ───────────── PROCESS VIDEO FUNCTION ─────────────
def process_and_return(video_path):
    """
    Accepts a video file path from Gradio, processes it, and returns the processed video path.
    """
    # video_path is already a string path, no need to read/write
    processed_path = process_video(video_path, OUTPUT_DIR, model, known_face_encodings, known_face_names)
    return processed_path

# ───────────── GRADIO INTERFACE ─────────────
iface = gr.Interface(
    fn=process_and_return,
    inputs=gr.File(file_types=[".mp4", ".avi", ".mov"], type="filepath"),  # <-- use filepath
    outputs=gr.Video(label="Processed Video"),
    title="🎥 Face Detection + Recognition (GPU)",
    description="Upload a video. YOLO runs on GPU and known faces are recognized automatically."
)


# ───────────── LAUNCH APP ─────────────
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
