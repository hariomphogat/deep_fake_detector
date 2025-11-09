import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2  # (opencv-python)
from mtcnn.mtcnn import MTCNN
import tempfile
import shutil

# --- Fix for MTCNN in Colab ---
tf.get_logger().setLevel('ERROR')

# 1. --- Import config ---
try:
    import config
except ImportError:
    print("Error: Could not import config.py.")
    sys.exit(1)

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def process_video_for_prediction(video_path, detector):
    """
    This is a complete pipeline to process one video for prediction.
    1. Extracts 30 frames
    2. Runs MTCNN to find/crop faces
    3. Resizes faces to 299x299
    4. Normalizes and stacks them into a (1, 30, 299, 299, 3) batch.
    """
    
    print("Processing video... This may take a moment.")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < config.SEQUENCE_LENGTH:
        print(f"Warning: Video is too short. Has {total_frames} frames, needs {config.SEQUENCE_LENGTH}.")
        # We can still try, but it's not ideal
    
    # Get evenly spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, config.SEQUENCE_LENGTH, dtype=int)
    
    # This will be our final (30, 299, 299, 3) array
    video_sequence = np.zeros(
        (config.SEQUENCE_LENGTH, config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE, 3), 
        dtype=np.float32
    )
    
    frames_processed = 0
    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            # --- MTCNN Face Detection ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(frame_rgb)
            
            if results:
                x1, y1, width, height = results[0]['box']
                x1, y1 = max(0, x1 - int(width*0.1)), max(0, y1 - int(height*0.1))
                x2, y2 = min(frame.shape[1], x1 + int(width*1.2)), min(frame.shape[0], y1 + int(height*1.2))
                
                face_crop = frame[y1:y2, x1:x2]
                
                # Resize to model's expected input
                face_resized = cv2.resize(face_crop, (config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE))
                
                # Normalize (just like in training)
                normalized_face = face_resized / 255.0
                
                # Add to our sequence
                video_sequence[i] = normalized_face
                frames_processed += 1
            # else:
                # If no face found, the frame at video_sequence[i] remains all zeros
                # print(f"Warning: No face found in frame {frame_index}")

        except Exception as e:
            print(f"Warning: Error on frame {frame_index}: {e}")
            pass
            
    cap.release()
    
    if frames_processed < (config.SEQUENCE_LENGTH * 0.5): # e.g., < 15 frames
        print("Error: Could not detect faces in most of the video. Aborting.")
        return None
        
    print(f"Successfully processed {frames_processed} frames.")
    
    # Add the "batch" dimension
    # Shape becomes (1, 30, 299, 299, 3)
    return np.expand_dims(video_sequence, axis=0)


def main(video_to_test):
    """
    Main prediction function for the CNN-LSTM model.
    """
    print("--- Starting Video Model Prediction ---")

    # 1. Check if the video file exists
    if not os.path.exists(video_to_test):
        print(f"Error: Video file not found at {video_to_test}")
        return

    # 2. Load our best fine-tuned VIDEO model
    model_path = os.path.join(config.MODEL_DIR, "finetuned_video_model.h5")
    if not os.path.exists(model_path):
        print(f"Error: Fine-tuned model not found at {model_path}")
        return
        
    print("Loading fine-tuned video model...")
    model = tf.keras.models.load_model(model_path)
    
    # 3. Initialize MTCNN face detector
    print("Initializing MTCNN face detector...")
    detector = MTCNN()
    
    # 4. Process the video
    # This returns a ready-to-predict batch
    video_batch = process_video_for_prediction(video_to_test, detector)
    
    if video_batch is None:
        print("Prediction cancelled.")
        return

    # 5. Make a prediction
    print("Model is making a prediction...")
    prediction_prob = model.predict(video_batch)[0][0]
    
    # 6. Interpret the result
    # 'fake' = 0, 'real' = 1
    
    if prediction_prob > 0.5:
        label = 'REAL'
        confidence = prediction_prob * 100
    else:
        label = 'FAKE'
        confidence = (1 - prediction_prob) * 100
    
    print("\n--- FINAL PREDICTION RESULT ---")
    print(f"Model prediction: This video is {label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"(Raw score: {prediction_prob:.4f} -> 0=Fake, 1=Real)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]
        main(VIDEO_PATH)
    else:
        print("--- Error: No video file path provided. ---")
        print("Please run this script from the command line like this:")
        print("python src/predict_video_model.py /path/to/your/video.mp4")