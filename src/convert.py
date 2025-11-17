import tensorflow as tf
import os

# Use relative import for use as a module
try:
    from . import config
except ImportError:
    import config

# --- Configuration ---
# 1. Set the path to your existing .h5 model file
h5_model_path = os.path.join(config.MODEL_DIR, "cnn_lstm_video_model.h5")

# 2. Set the desired path for the new .keras model file
keras_model_path = os.path.join(config.MODEL_DIR, "video_model_v1.keras")
# ---------------------

print(f"Loading model from: {h5_model_path}...")

try:
    # 1. Load the model from the .h5 file
    model = tf.keras.models.load_model(h5_model_path, compile=False)
    print("Model loaded successfully.")

    # 2. Save the model in the .keras format
    # TensorFlow automatically detects the format from the .keras extension
    print(f"Saving model to: {keras_model_path}...")
    model.save(keras_model_path)
    
    print("-" * 30)
    print("✅ Conversion Successful!")
    print(f"New model saved at: {keras_model_path}")
    print("-" * 30)

except FileNotFoundError:
    print(f"ERROR: The file '{h5_model_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("\nIf your model has custom layers or functions, you may need to register them.")
    print("See the 'Handling Custom Objects' section below.")