import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2  # (opencv-python)
from mtcnn.mtcnn import MTCNN

# Use relative import for use as a module
try:
    from . import config
except ImportError:
    import config

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')


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
    
    frame_indices = np.linspace(0, total_frames - 1, config.SEQUENCE_LENGTH, dtype=int)
    
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


def get_video_prediction(video_path, video_model, detector):

    # 1. Process the video using the helper function
    #    We pass the pre-loaded detector to it.
    video_batch = process_video_for_prediction(video_path, detector)
    
    # 2. Handle processing failure
    if video_batch is None:
        print("Video processing failed, returning error.")
        return {
            "prediction": "Error",
            "confidence": 0.0,
            "detail": "Could not process video or detect faces."
        }

    # 3. Make a prediction
    print("Model is making a prediction...")
    prediction_prob = video_model.predict(video_batch)[0][0]
    
    # 4. Interpret the result
    # 'fake' = 0, 'real' = 1
    if prediction_prob > 0.5:
        label = 'REAL'
        confidence = prediction_prob * 100
    else:
        label = 'FAKE'
        confidence = (1 - prediction_prob) * 100
    
    print("Prediction successful.")
    
    # 5. Return the result dictionary
    return {
        "prediction": label,
        "confidence": float(confidence),
        "raw_score": float(prediction_prob)
    }
