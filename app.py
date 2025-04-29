
import streamlit as st
from deepface import DeepFace
import requests
from PIL import Image
from gtts import gTTS
import os
import cv2
import numpy as np

# Constants
KNOWN_FOLDER = "known_faces"
ESP32_SERVER_URL = "https://esp32-upload-server.onrender.com"
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
MATCH_THRESHOLD = 0.3  # Strict threshold to prevent false matches

# Streamlit setup
st.set_page_config(page_title="Second Eye - Enhanced Recognition", layout="centered")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Face Recognition"])

# Fetch image from ESP32
def get_latest_image():
    try:
        r = requests.get(f"{ESP32_SERVER_URL}/latest")
        if r.status_code != 200:
            return None
        filename = r.json()["filename"]
        return f"{ESP32_SERVER_URL}/uploads/{filename}"
    except:
        return None

# Preprocess image for better recognition
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    gamma = 1.5
    lookUpTable = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(enhanced, lookUpTable)
    final_img = cv2.cvtColor(gamma_corrected, cv2.COLOR_GRAY2BGR)

    output_path = "preprocessed.jpg"
    cv2.imwrite(output_path, final_img)
    return output_path

# Detect faces
def is_face_detected(image_path):
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND
        )
        return len(faces) > 0
    except:
        return False

# Compare with known faces with strict control
def compare_with_known_faces(unknown_img_path):
    for filename in os.listdir(KNOWN_FOLDER):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        known_img_path = os.path.join(KNOWN_FOLDER, filename)
        try:
            result = DeepFace.verify(
                img1_path=unknown_img_path,
                img2_path=known_img_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )
            distance = result["distance"]
            if result["verified"] and distance < MATCH_THRESHOLD:
                return filename.split('.')[0]
        except Exception as e:
            print(f"Comparison failed with {filename}: {e}")
    return None

# Main UI
if page == "Face Recognition":
    st.title("Second-Eye 👁️👁️ Enhanced Face Recognition 🔎")

    if st.button("Check for New Image"):
        image_url = get_latest_image()
        if image_url:
            st.image(image_url, caption="Captured Image", use_container_width=True)

            response = requests.get(image_url)
            with open("latest.jpg", "wb") as f:
                f.write(response.content)

            processed_img_path = preprocess_image("latest.jpg")

            if is_face_detected(processed_img_path):
                match = compare_with_known_faces(processed_img_path)
                if match:
                    st.success(f"✅ Match found: {match}")
                    tts = gTTS(f"Match found: {match}")
                else:
                    st.error("❌ No match found")
                    tts = gTTS("No match found")
            else:
                st.warning("😕 No face detected in the image.")
                tts = gTTS("No face detected")

            tts.save("result.mp3")
            st.audio("result.mp3", autoplay=True)
        else:
            st.warning("No image found on ESP32 server")
