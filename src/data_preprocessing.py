import cv2
from mtcnn.mtcnn import MTCNN
import os
import sys
from tqdm import tqdm # Our progress bar library!
import warnings

# --- Suppress TensorFlow & MTCNN warnings ---
# This just quiets down the console output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
# --- End Warning Suppression ---

#
# 1. IMPORT FROM OUR CONFIG FILE
#
# This is the "best practice" part. We import all our paths 
# and settings from the single config.py file.
#
try:
    import config
except ImportError:
    print("Error: Could not import config.py.")
    print("Make sure it's in the 'src/' directory.")
    sys.exit(1)

def load_test_list(filepath):
    """
    Loads the official test file list into a set for fast lookup.
    
    The file format is:
    1/id0_0002.mp4
    1/id0_0006.mp4
    ...
    0/id2_0001.mp4
    
    We only care about the video filename (e.g., "id0_0002.mp4").
    The "1/" (fake) or "0/" (real) prefix in the list confirms the label.
    """
    test_videos = set()
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Get the part after the slash (e.g., "1/id0_0002.mp4" -> "id0_0002.mp4")
                filename = line.strip().split('/')[-1]
                test_videos.add(filename)
    except FileNotFoundError:
        print(f"Error: Test list file not found at: {filepath}")
        sys.exit(1)
    
    print(f"Loaded {len(test_videos)} videos into the test set.")
    return test_videos

def create_directories():
    """
    Creates all the necessary output directories defined in our config.
    The 'exist_ok=True' parameter prevents errors if the folders already exist.
    """
    print("Creating output directories...")
    os.makedirs(config.TRAIN_REAL_DIR, exist_ok=True)
    os.makedirs(config.TRAIN_FAKE_DIR, exist_ok=True)
    os.makedirs(config.TEST_REAL_DIR, exist_ok=True)
    os.makedirs(config.TEST_FAKE_DIR, exist_ok=True)
    print("Directories created/verified.")

def extract_faces(video_path, output_dir, video_filename, detector):
    """    
    Extracts, crops, and resizes faces from a single video file 
    and saves them to the specified output directory.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [Warning] Could not open video: {video_filename}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"  [Warning] Video has 0 frames: {video_filename}")
            return

        # Calculate a step to pick frames evenly, ensuring we don't just
        # get the first N frames.
        step = max(1, total_frames // config.FRAMES_PER_VIDEO)
        
        frame_num = 0
        faces_saved = 0

        while frame_num < total_frames and faces_saved < config.FRAMES_PER_VIDEO:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            
            ret, frame = cap.read()
            if not ret:
                frame_num += step
                continue
            
            # Convert frame to RGB for MTCNN
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            # This is the most time-consuming part
            faces = detector.detect_faces(frame_rgb)
            
            if faces:
                # Get the first face
                face = faces[0]
                x, y, w, h = face['box']

                # Make sure coordinates are not negative
                x, y = abs(x), abs(y)

                # Crop the face
                face_crop = frame[y : y+h, x : x+w]
                
                if face_crop.size == 0:
                    frame_num += step
                    continue
                
                # Resize to our standard size
                resized_face = cv2.resize(face_crop, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                
                # Save the image
                save_name = f"{video_filename}_frame_{frame_num}.jpg"
                save_path = os.path.join(output_dir, save_name)
                cv2.imwrite(save_path, resized_face)
                
                faces_saved += 1
                
            frame_num += step

    except Exception as e:
        print(f"  [Error] Processing {video_filename}: {e}")
    finally:
        if cap.isOpened():
            cap.release()
            
    return faces_saved

def process_all_videos(detector):
    """
    Orchestrates the entire preprocessing pipeline.
    """
    
    # Load the set of videos that belong in the "test" set
    test_set = load_test_list(config.TEST_LIST_FILE)
    
    # 1. --- Process REAL Videos ---
    # We combine 'Celeb-real' and 'Youtube-real' into one list
    real_video_dirs = [config.CELEB_REAL_DIR, config.YOUTUBE_REAL_DIR]
    
    real_video_paths = []
    for dir in real_video_dirs:
        for filename in os.listdir(dir):
            if filename.endswith('.mp4'):
                real_video_paths.append(os.path.join(dir, filename))
                
    print(f"\nFound {len(real_video_paths)} real videos. Processing...")
    
    # Use tqdm for a nice progress bar
    for video_path in tqdm(real_video_paths, desc="Processing Real Videos"):
        filename = os.path.basename(video_path)
        
        # Decide if it's train or test
        if filename in test_set:
            output_dir = config.TEST_REAL_DIR
        else:
            output_dir = config.TRAIN_REAL_DIR
            
        # Extract faces
        extract_faces(video_path, output_dir, os.path.splitext(filename)[0], detector)

    # 2. --- Process FAKE Videos ---
    fake_video_paths = []
    for filename in os.listdir(config.CELEB_FAKE_DIR):
        if filename.endswith('.mp4'):
            fake_video_paths.append(os.path.join(config.CELEB_FAKE_DIR, filename))
            
    print(f"\nFound {len(fake_video_paths)} fake videos. Processing...")
    
    # Use tqdm for a nice progress bar
    for video_path in tqdm(fake_video_paths, desc="Processing Fake Videos"):
        filename = os.path.basename(video_path)
        
        # Decide if it's train or test
        if filename in test_set:
            output_dir = config.TEST_FAKE_DIR
        else:
            output_dir = config.TRAIN_FAKE_DIR
            
        # Extract faces
        extract_faces(video_path, output_dir, os.path.splitext(filename)[0], detector)

#
# This is the "entry point" of our script
#
if __name__ == "__main__":
    print("--- DeepFake Detector: Data Preprocessing ---")
    
    # 1. Create all output folders
    create_directories()
    
    # 2. Initialize the MTCNN detector
    # We initialize it ONCE here and pass it to the functions.
    # This is much more efficient than creating one for every video.
    print("Initializing MTCNN face detector (this may take a moment)...")
    try:
        mtcnn_detector = MTCNN()
        print("MTCNN detector initialized.")
    except Exception as e:
        print(f"Fatal Error: Could not initialize MTCNN.")
        print("Please ensure TensorFlow is installed correctly.")
        print(f"Error details: {e}")
        sys.exit(1)

    # 3. Run the main processing loop
    process_all_videos(mtcnn_detector)
    
    print("\n--- Preprocessing Complete! ---")
    print(f"Your processed frames are now in: {config.PROCESSED_DATA_DIR}")