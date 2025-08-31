import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from feature_extractor import extract_features
from PIL import Image

st.title("Biomedical Sound & Face Mask Classification App")

st.header("Part 1: Heart/Lung Sound Classification")
uploaded_audio = st.file_uploader("Upload a .wav file", type=["wav"], key="audio")

if uploaded_audio is not None:
    with open("model/heart_lung_model.pkl", "rb") as f:
        audio_model = pickle.load(f)
    
    features = extract_features(uploaded_audio)  
    features = features.reshape(1, -1)
    
    pred = audio_model.predict(features)
    st.success(f"Prediction: {'Normal' if pred[0]==0 else 'Abnormal'}")

st.header("Part 2: Face Mask Detection")
uploaded_image = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], key="image")

if uploaded_image is not None:
    mask_model = load_model("model/best_cnn_model.keras")
    
    img = Image.open(uploaded_image).convert("RGB")
    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred_mask = mask_model.predict(img_array)
    label = "Mask" if np.argmax(pred_mask) == 0 else "No Mask"
    st.success(f"Prediction: {label}")
    
    st.image(img, caption="Uploaded Image", use_column_width=True)