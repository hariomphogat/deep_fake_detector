import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
from mtcnn.mtcnn import MTCNN  # <-- NEW IMPORT
import tensorflow as tf

# --- Fix for MTCNN in Colab ---
# MTCNN can have issues with TF2.x, this helps
tf.get_logger().setLevel('ERROR')

try:
    import config
except ImportError:
    print("Error: Could not import config.py. Make sure it's in the src/ directory.")
    exit(1)

def load_test_list():
    """Loads the list of test videos from the text file."""
    try:
        with open(config.TEST_LIST_FILE, 'r') as f:
            test_videos = [line.strip().split('/')[-1] for line in f]
        return set(test_videos)
    except FileNotFoundError:
        print(f"Error: Test list file not found at {config.TEST_LIST_FILE}")
        return set()

def extract_frames(video_path, output_folder, sequence_length, detector):
    """
    Extracts, face-detects, and crops 'sequence_length' 
    evenly spaced frames from a video.
    """
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  > Warning: Could not open video {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < sequence_length:
        print(f"  > Warning: Video {video_path} is too short ({total_frames} frames). Skipping.")
        return False
        
    frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
    os.makedirs(output_folder, exist_ok=True)
    
    frames_saved = 0
    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # --- NEW FACE DETECTION STEP ---
        try:
            # Convert from BGR (OpenCV) to RGB (MTCNN)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = detector.detect_faces(frame_rgb)
            
            if results:
                # Get the first and most confident face
                x1, y1, width, height = results[0]['box']
                # Add a 20% margin for safety
                x1, y1 = max(0, x1 - int(width*0.1)), max(0, y1 - int(height*0.1))
                x2, y2 = min(frame.shape[1], x1 + int(width*1.2)), min(frame.shape[0], y1 + int(height*1.2))
                
                # Crop the face from the *original* BGR frame
                face_crop = frame[y1:y2, x1:x2]
                
                # Resize the cropped face to our model's input size
                face_resized = cv2.resize(face_crop, (config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE))
                
                # Save the frame
                frame_filename = f"frame_{frames_saved:02d}.jpg"
                save_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(save_path, face_resized)
                frames_saved += 1
                
        except Exception as e:
            # Sometimes MTCNN fails on a weird frame
            print(f"  > Warning: Face detection failed for a frame in {video_path}. Error: {e}")
            pass
            
    cap.release()
    
    # Check if we actually saved enough frames
    if frames_saved < sequence_length * 0.8: # Allow for a few failures
        print(f"  > Warning: Only saved {frames_saved} frames for {video_path}. Skipping.")
        shutil.rmtree(output_folder) # Clean up partial folder
        return False
        
    return True

def main():
    """
    Main function to process all raw videos into sequence folders.
    """
    print("--- Starting Video Sequence Preprocessing (with Face Detection) ---")
    
    # 0. Clean old directory
    if os.path.exists(config.PROCESSED_SEQ_DIR):
        print(f"Removing old directory: {config.PROCESSED_SEQ_DIR}")
        shutil.rmtree(config.PROCESSED_SEQ_DIR)
        
    # 1. Load the list of test videos
    test_video_set = load_test_list()
    print(f"Loaded {len(test_video_set)} videos in the test set.")
    
    # 2. --- NEW: Initialize the Face Detector ---
    print("Initializing MTCNN face detector...")
    detector = MTCNN()
    print("Detector initialized.")
    
    data_sources = [
        ('real', config.CELEB_REAL_DIR),
        ('real', config.YOUTUBE_REAL_DIR),
        ('fake', config.CELEB_FAKE_DIR)
    ]
    
    video_count = 0
    
    for label, source_dir in data_sources:
        print(f"\nProcessing directory: {source_dir} (Label: {label})")
        if not os.path.exists(source_dir):
            print(f"  > Warning: Directory not found. Skipping.")
            continue
            
        # Use tqdm for a nice progress bar
        for video_file in tqdm(os.listdir(source_dir)):
            if not video_file.endswith(('.mp4', '.avi', '.mov')):
                continue
                
            video_path = os.path.join(source_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            
            if video_file in test_video_set:
                output_dir_base = config.TEST_SEQ_DIR
            else:
                output_dir_base = config.TRAIN_SEQ_DIR
                
            output_folder = os.path.join(output_dir_base, label, video_name)
            
            # Pass the detector to the function
            success = extract_frames(video_path, output_folder, config.SEQUENCE_LENGTH, detector)
            if success:
                video_count += 1

    print("\n--- Preprocessing Complete ---")
    print(f"Successfully processed {video_count} videos with face cropping.")
    print(f"New data is located in: {config.PROCESSED_SEQ_DIR}")

if __name__ == "__main__":
    main()