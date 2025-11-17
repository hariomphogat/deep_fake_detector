import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from mtcnn.mtcnn import MTCNN
import cv2

# We still need config for the image size, so this import is fine
try:
    from . import config
except ImportError:
    import config

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Initialize MTCNN detector globally (for efficiency)
print("Initializing MTCNN face detector...")
DETECTOR = MTCNN()
print("MTCNN initialized successfully.")

def detect_and_crop_face(image_path):
    """Detects and crops the largest face using MTCNN"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Detecting faces in the image...")
    faces = DETECTOR.detect_faces(img_rgb)
    
    if not faces:
        print("Warning: No faces detected!")
        return None
    
    if len(faces) > 1:
        print(f"Found {len(faces)} faces. Using the largest one.")
        faces = sorted(faces, key=lambda f: f['box'][2] * f['box'][3], reverse=True)
    
    face = faces[0]
    x, y, w, h = face['box']
    x, y = abs(x), abs(y)
    
    # CRITICAL FIX: No padding (matches training exactly)
    face_crop = img[y:y+h, x:x+w]
    
    if face_crop.size == 0:
        print("Error: Cropped face is empty")
        return None
    
    print(f"Face detected and cropped: {w}x{h} pixels")
    return face_crop

def preprocess_face(face_crop):
    """Preprocesses cropped face for model input"""
    resized_face = cv2.resize(face_crop, (config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE))
    resized_face_rgb = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
    normalized_face = resized_face_rgb.astype(np.float32) / 255.0
    img_batch = np.expand_dims(normalized_face, axis=0)
    return img_batch

def preprocess_image(img_path):
    """Complete preprocessing: detect, crop, and prepare"""
    face_crop = detect_and_crop_face(img_path)
    if face_crop is None:
        return None
    return preprocess_face(face_crop)

def get_image_prediction(image_path, model):
    print(f"\nProcessing image: {image_path}")
    
    if not os.path.exists(image_path):
        return {
            "prediction": None,
            "confidence": 0.0,
            "raw_score": 0.0,
            "error": f"File not found: {image_path}"
        }
    
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return {
            "prediction": None,
            "confidence": 0.0,
            "raw_score": 0.0,
            "error": "Failed to detect face in the image"
        }
    
    print("Running model prediction...")
    prediction_prob = model.predict(processed_image, verbose=0)[0][0]
    
    if prediction_prob > 0.5:
        label = 'REAL'
        confidence = prediction_prob * 100
    else:
        label = 'FAKE'
        confidence = (1 - prediction_prob) * 100
    
    print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
    
    return {
        "prediction": label,
        "confidence": float(confidence),
        "raw_score": float(prediction_prob),
        "error": None
    }

