# import streamlit as st
# import mysql.connector
# import requests
# import face_recognition
# import os
# import pyttsx3
# import threading
# import cv2
# import numpy as np
# from PIL import Image
# import io

# # Public Render Flask Server where ESP32 uploads the image
# ESP32CAM_URL = "https://esp32-upload-server.onrender.com/latest"
# SAVE_PATH = "captured_image.jpg"

# # MySQL Clever Cloud DB connection
# def connect_db():
#     return mysql.connector.connect(
#         host="b1fvdoqarhekhvzuhdcj-mysql.services.clever-cloud.com",
#         user="uulwfabkmrk4gxk2",
#         password="Indira@1943",  # 🔐 Replace with your real password
#         database="b1fvdoqarhekhvzuhdcj",
#         port=3306
#     )

# # Announce messages with TTS in a thread
# def announce(message):
#     def speak():
#         engine = pyttsx3.init()
#         engine.say(message)
#         engine.runAndWait()
#     threading.Thread(target=speak).start()

# # Encode face using face_recognition
# def encode_faces(image):
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_image)
#     face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
#     return face_encodings, face_locations

# # Compare captured encodings with DB encodings
# def recognize_faces(captured_encodings):
#     conn = connect_db()
#     cursor = conn.cursor()
#     cursor.execute("SELECT image_name, image_data FROM images")
#     data = cursor.fetchall()
#     cursor.close()
#     conn.close()

#     face_results = []

#     for captured_encoding in captured_encodings:
#         best_match = "Unknown"
#         min_distance = 0.6

#         for image_name, image_blob in data:
#             stored_image = np.array(Image.open(io.BytesIO(image_blob)))
#             stored_encodings, _ = encode_faces(stored_image)

#             for stored_encoding in stored_encodings:
#                 distance = face_recognition.face_distance([stored_encoding], captured_encoding)[0]
#                 if distance < min_distance:
#                     min_distance = distance
#                     best_match = image_name
        
#         face_results.append(best_match if best_match != "Unknown" else "Unknown Face")

#     return face_results

# # Draw boxes and labels on image
# def draw_bounding_boxes(image, face_locations, match_results):
#     for (top, right, bottom, left), name in zip(face_locations, match_results):
#         cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return image

# # Upload image to DB
# def upload_image_to_db(image, image_name):
#     conn = connect_db()
#     cursor = conn.cursor()
#     image_bytes = io.BytesIO()
#     image.save(image_bytes, format='JPEG')
#     image_data = image_bytes.getvalue()
    
#     cursor.execute("INSERT INTO images (image_name, image_data) VALUES (%s, %s)", (image_name, image_data))
#     conn.commit()
#     cursor.close()
#     conn.close()

# # --- Streamlit App UI ---
# st.title("Second Eye - Image Recognition System 👁️")

# page = st.sidebar.selectbox("Select Page", ["Home", "Supervisor", "User"])

# if page == "Home":
#     st.subheader("Live Recognition from ESP32-CAM")

#     try:
#         response = requests.get(ESP32CAM_URL)
#         if response.status_code == 200:
#             with open(SAVE_PATH, "wb") as f:
#                 f.write(response.content)

#             captured_image = cv2.imread(SAVE_PATH)
#             captured_encodings, face_locations = encode_faces(captured_image)

#             if captured_encodings:
#                 match_results = recognize_faces(captured_encodings)
#                 processed_image = draw_bounding_boxes(captured_image, face_locations, match_results)
#                 processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
#                 st.image(processed_pil, caption="Processed Image", use_column_width=True)

#                 for idx, name in enumerate(match_results, 1):
#                     if name == "Unknown Face":
#                         announce(f"Face {idx}: Not recognized")
#                         st.warning(f"Face {idx}: Not recognized")
#                     else:
#                         announce(f"Face {idx}: Match found with {name}")
#                         st.success(f"Face {idx}: Match found with {name}")
#             else:
#                 st.warning("No face detected.")
#                 announce("No face detected.")
#         else:
#             st.error("ESP32-CAM image fetch failed.")
#             announce("Failed to capture image.")
#     except Exception as e:
#         st.error(f"Error: {e}")
#         announce("Something went wrong.")

# elif page == "Supervisor":
#     st.subheader("Upload Known Face")
#     uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
#     name = st.text_input("Enter Name")

#     if st.button("Upload") and uploaded and name:
#         image = Image.open(uploaded)
#         upload_image_to_db(image, name)
#         st.success("Image uploaded successfully!")

# elif page == "User":
#     st.subheader("Stored Known Faces")
#     conn = connect_db()
#     cursor = conn.cursor()
#     cursor.execute("SELECT image_name, image_data FROM images")
#     images = cursor.fetchall()
#     cursor.close()
#     conn.close()

#     for image_name, blob in images:
#         image = Image.open(io.BytesIO(blob))
#         st.image(image, caption=image_name, use_column_width=True)




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
