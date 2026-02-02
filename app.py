import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from deep_translator import GoogleTranslator
from chatbot_data import BREED_DATA

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Cattle & Buffalo Breed Identification",
    page_icon="🐄",
    layout="centered"
)

# -------------------------------
# Session State Init
# -------------------------------
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------------------
# Translation Helper
# -------------------------------
def t(text, language):
    if language == "Hindi":
        return GoogleTranslator(source="en", target="hi").translate(text)
    return text

# -------------------------------
# Load Model
# -------------------------------
model = load_model("cattle_breed_model.h5")

class_names = [
    "Gir",
    "Jaffarabadi",
    "Murrah",
    "Red_Sindhi",
    "Sahiwal",
    "Tharparkar"
]

# -------------------------------
# Header
# -------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🐄 Cattle & Buffalo Breed Identification</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:gray;'>AI-based image recognition and advisory system for farmers</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------
# Language Selection
# -------------------------------
language = st.radio(
    "Language / भाषा",
    ("English", "Hindi"),
    horizontal=True
)

# -------------------------------
# Accuracy Info
# -------------------------------
st.info(t(
    "For best accuracy:\n"
    "• Upload a clear full-body image\n"
    "• Ensure good lighting\n"
    "• Avoid very close face-only images",
    language
))

# -------------------------------
# Image Input
# -------------------------------
st.markdown(f"### 📷 {t('Select Image Input Method', language)}")

option = st.radio(
    "",
    (t("Upload Image", language), t("Use Camera", language)),
    horizontal=True
)

if option == t("Upload Image", language):
    file = st.file_uploader(
        t("Click or Drag & Drop an Image (JPG / PNG / JPEG)", language),
        type=["jpg", "jpeg", "png"]
    )
    if file:
        st.session_state.uploaded_image = Image.open(file)
        st.session_state.chat = []  # reset chat only for new image

else:
    cam = st.camera_input("📸")
    if cam:
        st.session_state.uploaded_image = Image.open(cam)
        st.session_state.chat = []  # reset chat only for new image

# -------------------------------
# Prediction + Chatbot
# -------------------------------
if st.session_state.uploaded_image is not None:

    img = st.session_state.uploaded_image

    st.markdown("---")
    st.image(img, use_column_width=True)

    img = img.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)

    pred = model.predict(arr)
    idx = np.argmax(pred)
    confidence = pred[0][idx] * 100
    breed = class_names[idx]

    st.success(f"{t('Breed Identified', language)}: {breed}")
    st.write(f"{t('Confidence', language)}: {confidence:.2f}%")

    # -------------------------------
    # Chatbot
    # -------------------------------
    st.markdown("---")
    st.markdown(f"### 🤖 {t('Chat with Breed Assistant', language)}")

    if len(st.session_state.chat) == 0:
        st.session_state.chat.append({
            "role": "assistant",
            "content": f"Hello 👋 I have identified the breed as {breed}."
        })

        for sentence in BREED_DATA[breed]["info"]:
            st.session_state.chat.append({
                "role": "assistant",
                "content": sentence
            })

        st.session_state.chat.append({
            "role": "assistant",
            "content": (
                "What would you like to know next?\n"
                "Type breeding, vaccination, or nutrition."
            )
        })

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(
                t(msg["content"], language)
                if msg["role"] == "assistant"
                else msg["content"]
            )

    user_input = st.chat_input(t("Type your message...", language))

    if user_input:
        st.session_state.chat.append({
            "role": "user",
            "content": user_input
        })

        user_lower = user_input.lower()

        if "breed" in user_lower:
            reply = BREED_DATA[breed]["breeding"]
        elif "vaccin" in user_lower:
            reply = BREED_DATA[breed]["vaccine"]
        elif "nutri" in user_lower:
            reply = BREED_DATA[breed]["nutrition"]
        else:
            reply = (
                "I can help you with:\n"
                "• Breeding advice\n"
                "• Vaccination details\n"
                "• Nutrition guidance\n\n"
                "Please type breeding, vaccination, or nutrition."
            )

        st.session_state.chat.append({
            "role": "assistant",
            "content": reply
        })

        st.rerun()

