
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224

st.set_page_config(page_title="DermaVision AI", layout="centered")

st.title("DermaVision AI")
st.subheader("AI-Based Skin Lesion Classification System")

st.write(
    "Upload a dermoscopic skin lesion image. The AI model will predict the likely lesion category."
)

@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model("dermavision_skin_cancer_model.keras")

model = load_ai_model()

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

with open("class_names.json", "r") as f:
    class_names = json.load(f)

index_to_class = {int(v): k for k, v in class_indices.items()}

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    predicted_code = index_to_class[predicted_index]
    predicted_name = class_names[predicted_code]

    st.subheader("Prediction Result")
    st.write(f"**Predicted class:** {predicted_name}")
    st.write(f"**Class code:** {predicted_code}")
    st.write(f"**Confidence:** {confidence:.2%}")

    if predicted_code in ["mel", "bcc", "akiec"]:
        st.error("This may indicate a higher-risk lesion. Please consult a qualified dermatologist.")
    else:
        st.success("This appears lower risk, but medical confirmation is still recommended.")

    st.warning(
        "Disclaimer: This tool is for educational use only and does not replace professional medical diagnosis."
    )
