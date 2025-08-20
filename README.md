# 🎥 Face Detection & Recognition Video Processor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)](https://github.com/ultralytics/ultralytics)
[![Face Recognition](https://img.shields.io/badge/Face_Recognition-Dlib-orange)](https://github.com/ageitgey/face_recognition)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20App-yellow)](https://gradio.app/)

A Python-based **video processing application** that detects faces using **YOLOv8**, recognizes known faces using [`face_recognition`](https://github.com/ageitgey/face_recognition), and provides a **GPU-accelerated Gradio interface** for uploading and processing videos.

---

## 🔹 Features

- 🚀 Real-time face detection with **YOLOv8**
- 🧑‍🤝‍🧑 Recognizes known faces from a folder of images
- ⚡ GPU acceleration (CUDA) for faster inference
- 🎬 Annotated video output with bounding boxes and labels
- 🌐 User-friendly **Gradio web interface**
- ⚙️ Configurable tolerance & recognition model
- 🎥 Supports `.mp4`, `.avi`, `.mov` formats

---

## 🔹 Folder Structure

face-detection-recognition/

├── app.py # Main Gradio app

├── config.py # Configuration (paths, output folder)

├── face_utils.py # Face loading & recognition functions

├── video_processor.py # Video processing logic

├── known/ # Known faces folder

│ └── obama/ # Person subfolder with images

│ ├── obama (1).jpg

│ ├── obama (2).jpg

│ └── ...

├── face.pt # YOLOv8 model for face detection

├── output/ # Processed video output folder

├── requirements.txt # Required Python packages

└── .env # Environment variables

yaml
Copy
Edit

---

## 🔹 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd face-detection-recognition
Create a virtual environment (recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set up environment variables

Create a .env file:

env
Copy
Edit
FACE_MODEL_PATH=face.pt
KNOWN_PEOPLE_DIR=known
OUTPUT_DIR=output
🔹 Usage
Run the Gradio App
bash
Copy
Edit
python app.py
The app will start a local web server.

Open the provided link in your browser.

Upload a video file and get the processed video with recognized faces.

Process Video via Script
python
Copy
Edit
from video_processor import process_video
from face_utils import load_known_faces
from config import KNOWN_PEOPLE_DIR, OUTPUT_DIR, FACE_MODEL_PATH
from ultralytics import YOLO
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO(FACE_MODEL_PATH).to(device)

known_face_encodings, known_face_names = load_known_faces(KNOWN_PEOPLE_DIR)

video_path = "input_video.mp4"
output_path = process_video(video_path, OUTPUT_DIR, model, known_face_encodings, known_face_names)
print("Processed video saved at:", output_path)
🔹 Notes / Tips
✅ Organize known faces in subfolders, one per person

✅ Use YOLOv8 face.pt model (pretrained or custom-trained)

✅ For best accuracy, use cnn model with GPU

✅ Adjust tolerance in recognition if results are not accurate

✅ Output videos are saved in the folder defined in .env

🔹 Requirements
nginx
Copy
Edit
ultralytics
torch
opencv-python
face_recognition
numpy
gradio
python-dotenv
Install all dependencies via:

bash
Copy
Edit
pip install -r requirements.txt
🔹 Screenshots (Example)
📌 Add your screenshots here after running the app!
Example:

bash
Copy
Edit
/screenshots/
   
   ├── upload_page.png
   
   ├── processing.png
   
   └── output_video.png

🔹 Acknowledgements
Ultralytics YOLOv8

face_recognition

Gradio

