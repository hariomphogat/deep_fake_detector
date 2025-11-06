import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2  # (opencv-python)
import tempfile
import shutil

# 1. --- Import config ---
try:
    # This assumes the script is in 'src/' and 'config.py' is also in 'src/'
    import config
except ImportError:
    print("Error: Could not import config.py.")
    print("Make sure this script is in the 'src/' directory.")
    sys.exit(1)

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def extract_frames(video_path, temp_dir):
    """
    Extracts frames from a video, saving them to a temporary directory.
    Frames are sampled evenly from the video.
    """
    print(f"Extracting {config.FRAMES_PER_VIDEO} frames from video...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_extract = config.FRAMES_PER_VIDEO
    
    # If video has fewer frames than we want, just extract all of them
    if total_frames < frames_to_extract:
        frames_to_extract = total_frames
        
    # Calculate indices of frames to sample evenly
    if frames_to_extract == 0:
        print("Error: Video has 0 frames.")
        return []
        
    frame_indices = np.linspace(0, total_frames - 1, frames_to_extract, dtype=int)
    
    saved_frame_paths = []
    
    for i, frame_index in enumerate(frame_indices):
        # Set the video's current frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame at index {frame_index}.")
            continue
            
        # Save the frame to our temporary directory
        frame_filename = f"frame_{i:03d}.jpg"
        save_path = os.path.join(temp_dir, frame_filename)
        cv2.imwrite(save_path, frame)
        saved_frame_paths.append(save_path)

    cap.release()
    print(f"Successfully extracted {len(saved_frame_paths)} frames.")
    return saved_frame_paths


def preprocess_image(img_path):
    """
    Loads and preprocesses a single image for the model.
    (Identical to the function in predict.py)
    """
    img = image.load_img(img_path, target_size=(config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def main(video_to_test):
    """
    Main prediction function for a video file.
    """
    print("--- Starting Video Prediction ---")

    # 1. Check if the video file exists
    if not os.path.exists(video_to_test):
        print(f"Error: Video file not found at {video_to_test}")
        return

    # 2. Load our fine-tuned model
    model_path = os.path.join(config.MODEL_DIR, "finetuned_model.h5")
    if not os.path.exists(model_path):
        print(f"Error: Fine-tuned model not found at {model_path}")
        return
        
    print("Loading fine-tuned model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded.")

    temp_dir = None
    try:
        # 3. Create a unique temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {temp_dir}")
        
        # 4. Extract frames from the video
        frame_paths = extract_frames(video_to_test, temp_dir)
        
        if not frame_paths:
            print("No frames were extracted. Aborting.")
            return

        # 5. Run prediction on each frame
        print("Running prediction on extracted frames...")
        predictions = []
        for frame_path in frame_paths:
            processed_image = preprocess_image(frame_path)
            # Get the raw probability (0=Fake, 1=Real)
            pred_prob = model.predict(processed_image, verbose=0)[0][0]
            predictions.append(pred_prob)

        # 6. Aggregate results
        if not predictions:
            print("Prediction failed for all frames.")
            return

        # Calculate the average probability
        avg_prob = np.mean(predictions)
        
        # Count frames
        fake_frames = np.sum(np.array(predictions) < 0.5)
        real_frames = len(predictions) - fake_frames
        
        if avg_prob > 0.5:
            label = 'REAL'
            confidence = avg_prob * 100
        else:
            label = 'FAKE'
            confidence = (1 - avg_prob) * 100

        # 7. Print the final verdict
        print("\n--- VIDEO PREDICTION RESULT ---")
        print(f"Final Verdict: This video is {label}")
        print(f"Confidence: {confidence:.2f}%")
        print("---")
        print(f"Analysis: {fake_frames} / {len(predictions)} frames detected as FAKE.")
        print(f"Analysis: {real_frames} / {len(predictions)} frames detected as REAL.")
        print(f"(Average Score: {avg_prob:.4f} -> 0=Fake, 1=Real)")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # 8. --- CLEANUP (This runs no matter what) ---
        if temp_dir and os.path.exists(temp_dir):
            print("\nCleaning up temporary frame directory...")
            try:
                # Recursively delete the entire directory
                shutil.rmtree(temp_dir)
                print(f"Successfully removed: {temp_dir}")
            except Exception as e:
                print(f"Error cleaning up {temp_dir}: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]
        main(VIDEO_PATH)
    else:
        print("--- Error: No video file path provided. ---")
        print("Please run this script from the command line like this:")
        print("python src/predict_video.py /path/to/your/video.mp4")