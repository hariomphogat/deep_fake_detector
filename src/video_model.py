import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.applications import Xception

# Import our settings (SEQUENCE_LENGTH, TARGET_IMAGE_SIZE)
try:
    import config
except ImportError:
    print("Error: Could not import config.py. Make sure it's in the src/ directory.")
    exit(1)

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_video_model():
    """
    Builds the CNN-LSTM video model.
    
    This model consists of two parts:
    1. Encoder (CNN): A pre-trained Xception model to extract
       features from each frame.
    2. Decoder (LSTM): An LSTM network to analyze the
       sequence of features over time.
    """
    print("Building new CNN-LSTM video model...")

    # --- 1. The "Encoder" (Feature Extractor) ---
    
    # Load the Xception base, but WITHOUT the top classification layers
    # `pooling='avg'` adds a GlobalAveragePooling2D layer, turning the
    # 10x10x2048 feature map into a flat (2048,) vector.
    base_model = Xception(
        weights='imagenet', 
        include_top=False, 
        input_shape=(config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE, 3),
        pooling='avg'
    )
    
    # Freeze the base model. We will only train the new LSTM head first.
    base_model.trainable = False
    
    # --- 2. The Full Model (Encoder + Decoder) ---
    
    # Define the input shape for the *video*
    # (batch_size is implicit)
    # Shape: (30, 299, 299, 3) -> (frames, height, width, channels)
    video_input = Input(shape=(
        config.SEQUENCE_LENGTH, 
        config.TARGET_IMAGE_SIZE, 
        config.TARGET_IMAGE_SIZE, 
        3
    ))
    
    # --- The Magic Layer: TimeDistributed ---
    # This layer "wraps" our Xception model.
    # It tells Keras: "Apply this 'base_model' to every
    # single frame (all 30) in the sequence."
    #
    # Input to this layer: (Batch, 30, 299, 299, 3)
    # Output of this layer: (Batch, 30, 2048)
    
    encoded_frames = TimeDistributed(base_model)(video_input)
    
    # --- The "Decoder" (Temporal Analyzer) ---
    # Now we feed the sequence of 30 feature vectors into the LSTM.
    
    # We can stack LSTMs for more power.
    # `return_sequences=True` tells the first LSTM to output the
    # *full* 30-step sequence, not just the last step.
    x = LSTM(256, return_sequences=True)(encoded_frames)
    x = Dropout(0.5)(x)
    
    # The second LSTM only outputs the final step (the default behavior)
    # which summarizes the entire video.
    x = LSTM(128)(x)
    x = Dropout(0.5)(x)
    
    # --- Final Classification Head ---
    # A standard Dense head for the final "real/fake" decision
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    
    # Create the final model
    video_model = Model(video_input, output, name="cnn_lstm_video_model")
    
    return video_model

if __name__ == "__main__":
    # --- This is a quick test to see if the model builds ---
    print("Running a quick test to build the model...")
    try:
        model = build_video_model()
        
        print("\n--- Model Summary ---")
        model.summary()
        
        print("\nModel built successfully!")
        
    except Exception as e:
        print(f"\nModel build FAILED: {e}")
        print("Please check your config.py for TARGET_IMAGE_SIZE and SEQUENCE_LENGTH.")