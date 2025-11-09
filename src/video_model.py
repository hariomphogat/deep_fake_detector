import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.applications import Xception

# Import our settings (SEQUENCE_LENGTH, TARGET_IMAGE_SIZE)
try:
    from . import config
except ImportError:
    print("Error: Could not import config.py. Make sure it's in the src/ directory.")
    exit(1)

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_video_model():
    """
    Builds the CNN-LSTM video model.
    
    This model consists of two parts:
    1. Encoder (CNN): Instead of using a generic ImageNet Xception model, we are
    loading our OWN pre-trained 'finetuned_model.h5' and
    using ITS feature extractor. This extractor already knows how to spot deepfakes.
    2. Decoder (LSTM): An LSTM network to analyze the
       sequence of features over time.
    """
    print("Building new CNN-LSTM video model using 'finetuned_model.h5' as the encoder......")

    # --- 1. The "Encoder" (Feature Extractor: Pre-Trained Model) ---
    # Define the path to the best model
    finetuned_model_path = os.path.join(config.MODEL_DIR, "finetuned_model.h5")
    if not os.path.exists(finetuned_model_path):
        print(f"Error: Model not found at {finetuned_model_path}")
        print("Please run train.py and finetune.py first.")
        exit(1)
    
    # Load the full, fine-tuned image model
    full_image_model = tf.keras.models.load_model(finetuned_model_path)
    try:
        # Get the output of the pooling layer
        encoder_output = full_image_model.get_layer('global_average_pooling2d').output
        
        # Create a new model using the same input as the full model,
        # but outputting from the pooling layer
        base_model = Model(
            inputs=full_image_model.input, 
            outputs=encoder_output,
            name="finetuned_encoder"
        )
        
        print("Successfully loaded and rebuilt encoder from 'finetuned_model.h5'.")
        
    except ValueError:
        print("Error: Could not find layer 'global_average_pooling2d' in your saved model.")
        print("Falling back to ImageNet weights.")
        # This is a fallback, just in case
        base_model = Xception(
            weights='imagenet', 
            include_top=False, 
            input_shape=(config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE, 3),
            pooling='avg'
        )

    # Freeze this base model. Its weights are already perfect.
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
    # Apply our custom, fine-tuned 'base_model' to every
    # single frame (all 30) in the sequence."
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