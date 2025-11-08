import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pandas as pd
import random

# Import our settings (SEQUENCE_LENGTH, TARGET_IMAGE_SIZE)
try:
    import config
except ImportError:
    print("Error: Could not import config.py. Make sure it's in the src/ directory.")
    exit(1)

class VideoDataGenerator(tf.keras.utils.Sequence):
    """
    This is a Keras data generator for loading video sequences.
    It loads pre-processed sequences of frames from disk,
    preprocesses them, and yields them in batches for model training/testing.
    """
    
    def __init__(self, data_dir, batch_size, sequence_length,
                 img_size, shuffle=True):
        """
        Initialization
        :param data_dir: Path to the directory (e.g., .../train)
        :param batch_size: Size of each batch
        :param sequence_length: Number of frames per video
        :param img_size: Target image size (e.g., 299)
        :param shuffle: Whether to shuffle data at each epoch
        """
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.shuffle = shuffle
        
        # Build the dataframe that maps video folders to labels
        self.df = self.__create_video_dataframe()
        
        # This will be called at the end of each epoch
        self.on_epoch_end()

    def __create_video_dataframe(self):
        """
        Scans the data_dir and creates a DataFrame of
        [video_folder_path, label].
        """
        print(f"Creating video dataframe from: {self.data_dir}")
        video_paths = []
        labels = []
        
        # Walk through the data directory (train or test)
        for label in ['real', 'fake']:
            class_dir = os.path.join(self.data_dir, label)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue
                
            # Each folder inside is a video
            for video_folder in os.listdir(class_dir):
                video_folder_path = os.path.join(class_dir, video_folder)
                if os.path.isdir(video_folder_path):
                    video_paths.append(video_folder_path)
                    labels.append(label)
                    
        print(f"Found {len(video_paths)} videos.")
        return pd.DataFrame({'filepath': video_paths, 'label': labels})

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        
        # Get the video_folder_paths for this batch
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_df = self.df.iloc[start_idx:end_idx]
        
        # --- Initialize our batch arrays ---
        # X is our video data
        # (batch, seq_len, height, width, channels)
        X = np.empty((self.batch_size, self.sequence_length, self.img_size, self.img_size, 3), dtype=np.float32)
        
        # y is our labels
        y = np.empty((self.batch_size), dtype=int)
        
        
        # We must use 'enumerate' to get a batch index from 0 to 15 for a batch size of 16
        for i, (original_index, row) in enumerate(batch_df.iterrows()):
            # Get the path to the video folder
            video_folder_path = row['filepath']

            # Load the sequence of frames
            # 'i' is now our clean batch index (0, 1, 2...)
            X[i,] = self.__load_video_frames(video_folder_path)

            # Store the label (0 for fake, 1 for real)
            y[i] = 1 if row['label'] == 'real' else 0
            
        return X, y

    def on_epoch_end(self):
        """Shuffles the DataFrame at the end of each epoch."""
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __load_video_frames(self, video_folder_path):
        """
        Loads, preprocesses, and returns a sequence of frames
        from a single video folder.
        """
        
        # Initialize an empty array for this video's frames
        video_frames = np.empty((self.sequence_length, self.img_size, self.img_size, 3), dtype=np.float32)
        
        # Get all frame paths in the folder and sort them
        all_frames = sorted([f for f in os.listdir(video_folder_path) if f.endswith('.jpg')])
        
        # Select 'sequence_length' frames
        if len(all_frames) >= self.sequence_length:
            # If we have enough frames, take the first 'n'
            frames_to_load = all_frames[:self.sequence_length]
        else:
            # If video is shorter (shouldn't happen with our pre-processing)
            # We'll just use the frames we have and pad with zeros
            frames_to_load = all_frames
            print(f"Warning: Video folder {video_folder_path} has < {self.sequence_length} frames.")
            
        for i, frame_name in enumerate(frames_to_load):
            frame_path = os.path.join(video_folder_path, frame_name)
            
            # Load the image
            # The frames were already resized during pre-processing,
            # but this is a good safety check.
            img = image.load_img(frame_path, target_size=(self.img_size, self.img_size))
            img_array = image.img_to_array(img)
            
            # Normalize (rescale)
            img_array = img_array / 255.0
            
            video_frames[i] = img_array
        
        # (If we didn't have enough frames, the remaining entries 
        #  in 'video_frames' will be all zeros, which is fine)
            
        return video_frames

if __name__ == "__main__":
    # --- This is a quick test to see if the generator works ---
    print("Running a quick test of the VideoDataGenerator...")
    
    # We'll use the new path from our config file
    test_gen = VideoDataGenerator(
        data_dir=config.TRAIN_SEQ_DIR,
        batch_size=4, # Small batch for testing
        sequence_length=config.SEQUENCE_LENGTH,
        img_size=config.TARGET_IMAGE_SIZE
    )
    
    # Try to get one batch
    try:
        X_batch, y_batch = test_gen[0]
        print("\n--- Generator Test SUCCESS ---")
        print(f"X batch shape: {X_batch.shape}")
        print(f"y batch shape: {y_batch.shape}")
        print(f"X batch data type: {X_batch.dtype}")
        print(f"Labels in batch: {y_batch}")
    except Exception as e:
        print(f"\n--- Generator Test FAILED ---")
        print(f"Error: {e}")
        print("Please check that your 'processed_sequences/train' directory is not empty and has the correct structure.")