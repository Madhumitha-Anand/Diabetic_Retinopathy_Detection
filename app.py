import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

MODEL_PATH = "model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?export=download&id=14hGLy9qvdPb5_XMYvD0yLmM9biRJwM_C"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

IMG_SIZE = 224
class_names = ['DR', 'No_DR']

st.title("🩺 Diabetic Retinopathy Detection")

uploaded_file = st.file_uploader("Upload retinal image", type=["jpg", "png", "jpeg"])

def predict(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    return class_names[np.argmax(pred)], np.max(pred)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    if st.button("Predict"):
        result, conf = predict(image)

        if result == "DR":
            st.error(f"⚠️ DR Detected ({conf:.2f})")
        else:
            st.success(f"✅ No DR ({conf:.2f})")

st.warning("⚠️ This is not a medical diagnosis.")