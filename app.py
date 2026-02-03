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
# Session State
# -------------------------------
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "image_id" not in st.session_state:
    st.session_state.image_id = None

if "chat" not in st.session_state:
    st.session_state.chat = []

if "breed" not in st.session_state:
    st.session_state.breed = None

# -------------------------------
# Translation
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
    "Gir", "Jaffarabadi", "Murrah",
    "Red_Sindhi", "Sahiwal", "Tharparkar"
]

# -------------------------------
# Header
# -------------------------------
st.markdown("<h1 style='text-align:center;'>🐄 Cattle & Buffalo Breed Identification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>AI-based advisory system for farmers</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# Language
# -------------------------------
language = st.radio(
    "Language / भाषा",
    ("English", "Hindi"),
    horizontal=True
)

# -------------------------------
# Info
# -------------------------------
st.info(t(
    "For best accuracy:\n"
    "• Upload a clear full-body cattle image\n"
    "• Avoid bikes, people, screenshots\n"
    "• Ensure good lighting",
    language
))

# -------------------------------
# Image Input
# -------------------------------
st.markdown("### 📷 Select Image Input Method")

option = st.radio(
    "Image Input",
    ("Upload Image", "Use Camera"),
    horizontal=True,
    label_visibility="collapsed"
)

uploaded = None

if option == "Upload Image":
    uploaded = st.file_uploader(
        "Upload JPG / PNG image",
        type=["jpg", "jpeg", "png"]
    )
else:
    uploaded = st.camera_input("Camera")

# -------------------------------
# Handle New Image Upload
# -------------------------------
if uploaded is not None:
    image_id = hash(uploaded.getvalue())

    if st.session_state.image_id != image_id:
        st.session_state.image_id = image_id
        st.session_state.uploaded_image = Image.open(uploaded)
        st.session_state.chat = []
        st.session_state.breed = None

# =========================================================
# After Image Upload
# =========================================================
if st.session_state.uploaded_image is not None:

    img = st.session_state.uploaded_image
    st.image(img, use_container_width=True)

    img = img.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)

    pred = model.predict(arr)
    idx = np.argmax(pred)
    confidence = float(pred[0][idx] * 100)

    breed = class_names[idx]
    st.session_state.breed = breed

    if confidence < 60:
        st.error(t(
            "This image does not appear to be a cow or buffalo. "
            "Please upload a clear cattle image.",
            language
        ))
        st.write(f"{t('Model Confidence', language)}: {confidence:.2f}%")
        st.stop()

    st.success(f"{t('Breed Identified', language)}: {breed}")
    st.write(f"{t('Confidence', language)}: {confidence:.2f}%")

    # -------------------------------
    # Chatbot
    # -------------------------------
    st.markdown("---")
    st.markdown("### 🤖 Chat with Breed Assistant")

    breed = st.session_state.breed

    # Initialize chat once
    if len(st.session_state.chat) == 0:
        st.session_state.chat.append({
            "role": "assistant",
            "content": f"I have identified the breed as {breed}."
        })

        for info in BREED_DATA[breed]["info"]:
            st.session_state.chat.append({
                "role": "assistant",
                "content": info
            })

        st.session_state.chat.append({
            "role": "assistant",
            "content": (
                "You can ask about:\n"
                "• breeding\n"
                "• vaccination\n"
                "• nutrition"
            )
        })

    # -------------------------------
    # Chat Input
    # -------------------------------
    user_input = st.chat_input(
        t("Type breeding, vaccination, or nutrition...", language)
    )

    if user_input:
        user_text = user_input.strip().lower()

        st.session_state.chat.append({
            "role": "user",
            "content": user_input
        })

        if user_text == "breeding":
            reply = BREED_DATA[breed]["breeding"]

        elif user_text == "vaccination":
            reply = BREED_DATA[breed]["vaccine"]

        elif user_text == "nutrition":
            reply = BREED_DATA[breed]["nutrition"]

        else:
            reply = (
                "Invalid input ❌\n\n"
                "Please type only:\n"
                "• breeding\n"
                "• vaccination\n"
                "• nutrition"
            )

        st.session_state.chat.append({
            "role": "assistant",
            "content": reply
        })

    # -------------------------------
    # Display Chat (LAST)
    # -------------------------------
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(
                t(msg["content"], language)
                if msg["role"] == "assistant"
                else msg["content"]
            )



