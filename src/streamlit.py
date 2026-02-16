import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
from PIL import Image
import cv2
import os
from ultralytics import YOLO

# ----------------- Model Loader -----------------
class ModelLoader:
    @staticmethod
    def load_heart_sound_model():
        path = os.path.join("model", "heart_lung_model.keras")
        if not os.path.exists(path):
            st.sidebar.error(f"❌ Heart sound model not found at {path}")
            return None
        try:
            model = tf.keras.models.load_model(path)
            st.sidebar.success("✅ Heart Sound Model Loaded!")
            return model
        except Exception as e:
            st.sidebar.error(f"❌ Error loading heart model: {e}")
            return None

    @staticmethod
    def load_face_mask_models():
        cnn_path = os.path.join("model", "best_cnn_model.keras")
        yolo_path = os.path.join("model", "yolov5su.pt")
        if not os.path.exists(cnn_path) or not os.path.exists(yolo_path):
            st.sidebar.error("❌ CNN or YOLO model not found!")
            return None, None
        try:
            cnn_model = tf.keras.models.load_model(cnn_path)
            yolo_model = YOLO(yolo_path)
            st.sidebar.success("✅ Face Mask Models Loaded!")
            return cnn_model, yolo_model
        except Exception as e:
            st.sidebar.error(f"❌ Error loading face mask models: {e}")
            return None, None

# ----------------- Feature Extractor -----------------
class FeatureExtractor:
    @staticmethod
    def extract_audio_features(audio_data, sr):
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mean = np.mean(audio_data)
        std = np.std(audio_data)
        max_val = np.max(audio_data)
        min_val = np.min(audio_data)
        features = np.concatenate([[mean, std, max_val, min_val], mfccs_mean])
        return features.reshape(1, -1)

    @staticmethod
    def preprocess_image(image, target_size=(224, 224)):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = cv2.resize(image, target_size)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image

# ----------------- Image Processor -----------------
class ImageProcessor:
    @staticmethod
    def draw_boxes(image, results, threshold=0.5):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        img = image.copy()
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if conf > threshold:
                    color = (0,255,0) if cls==0 else (0,0,255)
                    cv2.rectangle(img, (x1,y1),(x2,y2),color,2)
                    label = f"{'Mask' if cls==0 else 'No Mask'}: {conf:.2f}"
                    cv2.putText(img,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
        return img

# ----------------- Main App -----------------
class App:
    def __init__(self):
        st.set_page_config(page_title="Biomedical Classification", layout="wide")
        self.page = st.sidebar.radio("Select Task", ["Home","Heart Sound Classification","Face Mask Detection"])
        self.heart_model = None
        self.cnn_model = None
        self.yolo_model = None

    def load_models(self):
        if self.page=="Heart Sound Classification" and self.heart_model is None:
            self.heart_model = ModelLoader.load_heart_sound_model()
        elif self.page=="Face Mask Detection" and (self.cnn_model is None or self.yolo_model is None):
            self.cnn_model, self.yolo_model = ModelLoader.load_face_mask_models()

    def run(self):
        self.load_models()
        if self.page=="Home":
            st.title("🏥 Biomedical Classification System")
            st.write("Select a task from the sidebar to start!")
        elif self.page=="Heart Sound Classification":
            st.title("Heart Sound Classification")
            if self.heart_model is None:
                st.warning("⚠️ Heart sound model missing!")
                return
            uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
            if uploaded_file:
                st.audio(uploaded_file)
                if st.button("Classify"):
                    audio, sr = librosa.load(uploaded_file)
                    features = FeatureExtractor.extract_audio_features(audio,sr)
                    pred = self.heart_model.predict(features)[0]
                    idx = np.argmax(pred)
                    classes = ["Normal","Murmur","Extra Heart Sound","Artifact"]
                    st.success(f"Predicted Class: **{classes[idx]}**, Confidence: {pred[idx]:.2f}")

        elif self.page=="Face Mask Detection":
            st.title("Face Mask Detection")
            if self.cnn_model is None or self.yolo_model is None:
                st.warning("⚠️ Face mask models missing!")
                return
            uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
            if uploaded_file:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_column_width=True)
                if st.button("Detect"):
                    # CNN
                    cnn_input = FeatureExtractor.preprocess_image(img)
                    cnn_pred = self.cnn_model.predict(cnn_input)[0]
                    cnn_result = "Mask" if cnn_pred>0.5 else "No Mask"
                    cnn_conf = cnn_pred if cnn_pred>0.5 else 1-cnn_pred
                    # YOLO
                    results = self.yolo_model(np.array(img))
                    annotated = ImageProcessor.draw_boxes(img, results[0])
                    col1,col2 = st.columns(2)
                    with col1:
                        st.markdown("### CNN Results")
                        st.write(f"{cnn_result}, Confidence: {cnn_conf:.2f}")
                    with col2:
                        st.markdown("### YOLO Results")
                        st.image(annotated, use_column_width=True)

if __name__=="__main__":
    App().run()
