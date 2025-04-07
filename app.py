import streamlit as st
import requests
import face_recognition
from PIL import Image
from gtts import gTTS
import os
import io

KNOWN_FOLDER = "known_faces"
ESP32_SERVER_URL = "https://esp32-upload-server.onrender.com"
FLASK_UPLOAD_URL = "https://flask-upload-pzch.onrender.com/upload"
MAX_FILE_SIZE_MB = 3
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Face Recognition", "Upload Known Face"])

# Load known encodings
def load_known_encodings():
    encodings = []
    names = []
    for file in os.listdir(KNOWN_FOLDER):
        path = os.path.join(KNOWN_FOLDER, file)
        img = face_recognition.load_image_file(path)
        face_enc = face_recognition.face_encodings(img)
        if face_enc:
            encodings.append(face_enc[0])
            names.append(file.split('.')[0])
    return encodings, names

# Get latest image from ESP32 server
def get_latest_image():
    r = requests.get(f"{ESP32_SERVER_URL}/latest")
    if r.status_code != 200:
        return None
    filename = r.json()["filename"]
    return f"{ESP32_SERVER_URL}/uploads/{filename}"

# -------------------- PAGE 1: Face Recognition --------------------
if page == "Face Recognition":
    st.title("ESP32-CAM Face Recognition (face_recognition + dlib)")

    if st.button("Check for New Image"):
        image_url = get_latest_image()
        if image_url:
            st.image(image_url, caption="Captured Image", use_container_width=True)
            img_data = requests.get(image_url).content
            with open("latest.jpg", "wb") as f:
                f.write(img_data)

            unknown_image = face_recognition.load_image_file("latest.jpg")
            unknown_encodings = face_recognition.face_encodings(unknown_image)

            if unknown_encodings:
                known_encodings, known_names = load_known_encodings()
                results = face_recognition.compare_faces(known_encodings, unknown_encodings[0])
                if True in results:
                    index = results.index(True)
                    match = known_names[index]
                    st.success(f"✅ Match found: {match}")
                    tts = gTTS(f"Match found: {match}")
                else:
                    st.error("❌ No match found")
                    tts = gTTS("No match found")
            else:
                st.warning("😕 No face detected in the captured image.")
                tts = gTTS("No face detected")

            tts.save("result.mp3")
            st.audio("result.mp3", autoplay=True)
        else:
            st.warning("No image found on server.")

# -------------------- PAGE 2: Upload Known Face --------------------
elif page == "Upload Known Face":
    st.title("Upload New Known Face")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Upload to GitHub"):
            file_data = uploaded_file.getvalue()
            if len(file_data) > MAX_FILE_SIZE_BYTES:
                st.error(f"❌ File too large. Please upload a file under {MAX_FILE_SIZE_MB} MB.")
            else:
                try:
                    safe_filename = uploaded_file.name.replace(" ", "_")
                    files = {"file": (safe_filename, file_data)}
                    response = requests.post(FLASK_UPLOAD_URL, files=files, timeout=30)
                    if response.status_code == 201:
                        st.success("✅ Image uploaded to GitHub successfully.")
                    else:
                        st.error(f"❌ Upload failed. Status code: {response.status_code}\n{response.text}")
                except requests.exceptions.ChunkedEncodingError:
                    st.error("⚠️ Upload failed due to network or encoding error. Try again or use a smaller image.")
                except requests.exceptions.RequestException as e:
                    st.error(f"⚠️ Upload failed: {str(e)}")
