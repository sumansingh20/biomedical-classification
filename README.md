# Biomedical Classification System

This application provides two main functionalities:
1. Heart Sound Classification
2. Face Mask Detection

## Features

### Heart Sound Classification
- Upload and analyze heart sound recordings (.wav format)
- Classify into categories: Normal, Murmur, Extra Heart Sound, Artifact
- Get instant predictions with confidence scores

### Face Mask Detection
- Upload images for mask detection
- Uses both CNN and YOLO models for accurate detection
- Real-time predictions with bounding boxes
- Confidence scores for each detection

## Setup and Installation

### Prerequisites
- Python 3.9 or later
- pip package manager

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd biomedical-classification-app
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run src/streamlit.py
```

### Docker Installation

1. Build the Docker image:
```bash
docker build -t biomedical-classification-app .
```

2. Run the container:
```bash
docker run -p 8501:8501 biomedical-classification-app
```

The application will be available at http://localhost:8501

## Model Information

### Heart Sound Model
- Located in: `model/heart_lung_model.pkl`
- Input: .wav audio files
- Output: Classification with confidence score

### Face Mask Detection Models
- CNN Model: `model/best_cnn_model.keras`
- YOLO Model: `model/yolov5su.pt`
- Input: Image files (jpg, jpeg, png)
- Output: Mask detection with bounding boxes and confidence scores

## Usage

1. Open the application in your web browser
2. Select the desired task from the sidebar (Heart Sound or Face Mask Detection)
3. Upload your file (audio for heart sounds, image for mask detection)
4. Click the respective button to process
5. View results and predictions

## Project Structure

```
biomedical-classification-app/
├── src/
│   ├── streamlit.py          # Main Streamlit application
│   └── feature_extractor.py  # Feature extraction utilities
├── model/
│   ├── heart_lung_model.pkl  # Heart sound classification model
│   ├── best_cnn_model.keras  # Face mask CNN model
│   └── yolov5su.pt          # Face mask YOLO model
├── requirements.txt          # Project dependencies
├── Dockerfile               # Docker configuration
└── README.md               # Documentation
```

## Notes

- Ensure all model files are present in the `model` directory
- Use clear audio recordings for better heart sound classification
- For face mask detection, ensure images are well-lit and faces are clearly visible

## License

This project is licensed under the MIT License.
