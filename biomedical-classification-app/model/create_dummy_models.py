import tensorflow as tf
import numpy as np
from ultralytics import YOLO

# ---------------- Dummy Heart Sound Model ----------------
heart_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(17,), activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
heart_model.save("model/heart_lung_model.keras")
print("Dummy Heart Sound model created")

# ---------------- Dummy CNN Model ----------------
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
cnn_model.save("model/best_cnn_model.keras")
print("Dummy CNN model created")

# Use pre-trained YOLO small model
yolo_model = YOLO("yolov8n.pt")
yolo_model.save("model/yolov5su.pt")
print("Dummy YOLO model created")
