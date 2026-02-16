import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
import pickle

class ModelLoader:
    @staticmethod
    def load_heart_sound_model():
        try:
            model_path = "model/heart_lung_model.pkl"
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            st.sidebar.success("Heart Sound Model loaded successfully!")
            return model
        except Exception as e:
            st.sidebar.error(f"Error loading Heart Sound Model (pickle): {str(e)}")
            return None
    
    @staticmethod
    def load_face_mask_models():
        try:
            cnn_model = tf.keras.models.load_model("model/best_cnn_model.keras")
            yolo_model = YOLO("model/yolov5su.pt")
            st.sidebar.success("Face Mask Models loaded successfully!")
            return cnn_model, yolo_model
        except Exception as e:
            st.sidebar.error(f"Error loading Face Mask Models: {str(e)}")
            return None, None

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

class ImageProcessor:
    @staticmethod
    def draw_boxes(image, results, confidence_threshold=0.5):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            
        annotated_image = image.copy()
        
        if results.boxes is not None:
            boxes = results.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if conf > confidence_threshold:
                    color = (0, 255, 0) if cls == 0 else (0, 0, 255)  
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{'Mask' if cls == 0 else 'No Mask'}: {conf:.2f}"
                    cv2.putText(annotated_image, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_image

class App:
    def __init__(self):
        st.set_page_config(
            page_title="Biomedical Classification",
            page_icon="🏥",
            layout="wide"
        )
        self.setup_sidebar()
        self.heart_sound_model = None
        self.cnn_model = None
        self.yolo_model = None
        
    def setup_sidebar(self):
        st.sidebar.title("Navigation")
        self.page = st.sidebar.radio(
            "Select Task",
            ["Home", "Heart Sound Classification", "Face Mask Detection"]
        )
        
    def load_models(self):
        if self.page == "Heart Sound Classification" and self.heart_sound_model is None:
            self.heart_sound_model = ModelLoader.load_heart_sound_model()
            
        elif self.page == "Face Mask Detection" and (self.cnn_model is None or self.yolo_model is None):
            self.cnn_model, self.yolo_model = ModelLoader.load_face_mask_models()
    
    def show_home(self):
        st.title("🏥 Biomedical Classification System")
    
    def show_heart_sound_classification(self):
        st.title("Heart Sound Classification")
        
        if self.heart_sound_model is None:
            st.warning("Please ensure the heart sound model is available in the model directory.")
            return
        
        uploaded_file = st.file_uploader("Upload an audio file", type=['wav'])
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            if st.button("Classify"):
                with st.spinner("Processing audio..."):
                    audio_data, sr = librosa.load(uploaded_file)
                    features = FeatureExtractor.extract_audio_features(audio_data, sr)
                    
                    prediction = self.heart_sound_model.predict(features)
                    if hasattr(self.heart_sound_model, "predict_proba"):
                        proba = self.heart_sound_model.predict_proba(features)[0]
                        class_idx = np.argmax(proba)
                        confidence = proba[class_idx]
                    else:
                        class_idx = prediction[0]
                        confidence = None
                    classes = ['Normal', 'Murmur', 'Extra Heart Sound', 'Artifact']
                    result = classes[class_idx]
                    st.success("Classification complete!")
                    st.write(f"Predicted Class: **{result}**")
                    if confidence is not None:
                        st.write(f"Confidence: **{confidence:.2f}**")
    
    def show_face_mask_detection(self):
        st.title("Face Mask Detection")
        
        if self.cnn_model is None or self.yolo_model is None:
            st.warning(" Please ensure both CNN and YOLO models are available in the model directory.")
            return
        
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Detect"):
                with st.spinner("Processing image..."):
                    # CNN prediction
                    cnn_input = FeatureExtractor.preprocess_image(image)
                    cnn_pred = self.cnn_model.predict(cnn_input)[0]
                    cnn_result = "Mask" if cnn_pred > 0.5 else "No Mask"
                    cnn_conf = cnn_pred if cnn_pred > 0.5 else 1 - cnn_pred
                    
                    # YOLO detection
                    results = self.yolo_model(image)
                    annotated_image = ImageProcessor.draw_boxes(image, results[0])
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### CNN Model Results")
                        st.write(f"Prediction: **{cnn_result}**")
                        st.write(f"Confidence: **{cnn_conf:.2f}**")
                    
                    with col2:
                        st.markdown("### YOLO Model Results")
                        st.image(annotated_image, caption="Detected Faces", use_column_width=True)
    
    def run(self):
        self.load_models()
        
        if self.page == "Home":
            self.show_home()
        elif self.page == "Heart Sound Classification":
            self.show_heart_sound_classification()
        elif self.page == "Face Mask Detection":
            self.show_face_mask_detection()

if __name__ == "__main__":
    app = App()
    app.run()