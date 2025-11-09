import os

# --- Main Project Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
# PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
# MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
# RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ---Colab:Main Project Paths ---
# PROJECT_ROOT = "/content/deep_fake_detector"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
# MODEL_DIR = "/content/drive/MyDrive/models"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# --- Raw Data Paths ---
CELEB_REAL_DIR = os.path.join(RAW_DATA_DIR, "Celeb-real")
CELEB_FAKE_DIR = os.path.join(RAW_DATA_DIR, "Celeb-synthesis")
YOUTUBE_REAL_DIR = os.path.join(RAW_DATA_DIR, "Youtube-real")
TEST_LIST_FILE = os.path.join(RAW_DATA_DIR, "List_of_testing_videos.txt")

# --- Processed Data Paths ---
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")
TRAIN_REAL_DIR = os.path.join(TRAIN_DIR, "real")
TRAIN_FAKE_DIR = os.path.join(TRAIN_DIR, "fake")
TEST_REAL_DIR = os.path.join(TEST_DIR, "real")
TEST_FAKE_DIR = os.path.join(TEST_DIR, "fake")

# --- Processed Sequence Paths (for Video Model) ---
PROCESSED_SEQ_DIR = os.path.join(DATA_DIR, "processed_sequences")
TRAIN_SEQ_DIR = os.path.join(PROCESSED_SEQ_DIR, "train")
TEST_SEQ_DIR = os.path.join(PROCESSED_SEQ_DIR, "test")


# --- Preprocessing Parameters ---
IMAGE_SIZE = 224                # Our standard image size (224x224)
FRAMES_PER_VIDEO = 30           # Number of frames to extract from each video

# --- Model & Training Parameters ---
TARGET_IMAGE_SIZE = 299            # Size to which images are resized for model input
BATCH_SIZE = 64
VIDEO_BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 2e-3
SEQUENCE_LENGTH = 30

print("Configuration loaded:")
print(f"  Project Root: {PROJECT_ROOT}")
print(f"  Processed Data: {PROCESSED_DATA_DIR}")
print(f"  Test List File: {TEST_LIST_FILE}")