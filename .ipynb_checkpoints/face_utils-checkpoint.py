# face_utils.py
import os
import cv2
import face_recognition

def load_known_faces(known_dir, model_type="cnn"):
    """
    Loads known faces from a directory.
    Each subfolder should be named after the person and contain their images.
    """
    known_encodings = []
    known_names = []

    # Iterate over subfolders (each subfolder is a person)
    for name in os.listdir(known_dir):
        person_dir = os.path.join(known_dir, name)
        if not os.path.isdir(person_dir):
            continue  # skip files, only look at folders

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            ext = os.path.splitext(img_name)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load {img_path}")
                    continue
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                encs = face_recognition.face_encodings(rgb_img, model=model_type)
                if len(encs) > 0:
                    known_encodings.append(encs[0])
                    known_names.append(name)
                else:
                    print(f"No face found in {img_path}")

            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    print(f"Loaded {len(known_encodings)} known faces: {set(known_names)}")
    return known_encodings, known_names


def recognize_face(face_img, known_face_encodings, known_face_names, tolerance=0.6, model_type="cnn"):
    """
    Recognizes a single face image by comparing with known faces.
    Returns the name or 'Unknown' if no match.
    """
    rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img, model=model_type)

    if len(encodings) == 0:
        return "Unknown"

    face_encoding = encodings[0]
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
    name = "Unknown"

    if True in matches:
        matched_idxs = [i for i, m in enumerate(matches) if m]
        counts = {}
        for i in matched_idxs:
            counts[known_face_names[i]] = counts.get(known_face_names[i], 0) + 1
        name = max(counts, key=counts.get)

    return name
