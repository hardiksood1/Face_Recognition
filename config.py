import os
from dotenv import load_dotenv

load_dotenv()

FACE_MODEL_PATH = os.getenv("FACE_MODEL_PATH")
KNOWN_PEOPLE_DIR = os.getenv("KNOWN_PEOPLE_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)