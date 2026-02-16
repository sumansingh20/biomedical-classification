import os
import numpy as np
import tensorflow as tf
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ultralytics import YOLO

# ---------------- Heart Sound Model ----------------
print("\n===== Training Heart Sound Model =====")

DATA_PATH_AUDIO = "data"  
CLASSES_AUDIO = ["Normal", "Murmur", "Extra", "Artifact"]

X_audio = []
y_audio = []

for idx, class_name in enumerate(CLASSES_AUDIO):
    folder = os.path.join(DATA_PATH_AUDIO, class_name)
    if not os.path.exists(folder):
        print(f"Folder missing: {folder}")
        continue
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            audio, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            mean = np.mean(audio)
            std = np.std(audio)
            max_val = np.max(audio)
            min_val = np.min(audio)
            features = np.concatenate([[mean, std, max_val, min_val], mfccs_mean])
            X_audio.append(features)
            y_audio.append(idx)

X_audio = np.array(X_audio)
y_audio = tf.keras.utils.to_categorical(np.array(y_audio), num_classes=len(CLASSES_AUDIO))

X_train, X_test, y_train, y_test = train_test_split(X_audio, y_audio, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

heart_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(CLASSES_AUDIO), activation='softmax')
])

heart_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
heart_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

os.makedirs("model", exist_ok=True)
heart_model.save("model/heart_lung_model.keras")
print("✅ Heart Sound model saved at model/heart_lung_model.keras")

# ---------------- Face Mask CNN Model ----------------
print("\n===== Training Face Mask CNN Model =====")

DATA_PATH_IMAGE = "data_face_mask"  #
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15
)

train_gen = datagen.flow_from_directory(
    DATA_PATH_IMAGE,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_PATH_IMAGE,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_gen, validation_data=val_gen, epochs=20)

cnn_model.save("model/best_cnn_model.keras")
print("Face Mask CNN model saved at model/best_cnn_model.keras")

# ---------------- YOLO Model ----------------
print("\n===== YOLO Model Setup =====")

yolo_model = YOLO("yolov8n.pt")  
yolo_model.save("model/yolov5su.pt")
print(" YOLO model saved at model/yolov5su.pt")

print("\nAll models are ready! You can now run Streamlit app.")
