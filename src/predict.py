import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2 

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

def preprocess_image(img_path):
    """
    Loads and preprocesses a single image for the model.
    """
    # Load the image, resizing it to our model's expected input size
    img = image.load_img(img_path, target_size=(config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE))
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Rescale the pixel values (just like we did in training)
    img_array = img_array / 255.0
    
    # Expand the dimensions to create a "batch" of 1
    # Shape goes from (299, 299, 3) to (1, 299, 299, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch

def main(image_to_test):
    """
    Main prediction function.
    """
    print("--- Starting Single Image Prediction ---")

    # 1. Check if the image file exists
    if not os.path.exists(image_to_test):
        print(f"Error: Image file not found at {image_to_test}")
        return

    # 2. Load our BEST model (the fine-tuned one)
    model_path = os.path.join(config.MODEL_DIR, "finetuned_model.h5")
    if not os.path.exists(model_path):
        print(f"Error: Fine-tuned model not found at {model_path}")
        print("Run finetune.py first.")
        return
        
    print("Loading fine-tuned model...")
    model = tf.keras.models.load_model(model_path)
    
    # 3. Preprocess the image
    print(f"Processing image: {image_to_test}")
    processed_image = preprocess_image(image_to_test)
    
    # 4. Make a prediction
    # This will be a single number between 0 and 1
    prediction_prob = model.predict(processed_image)[0][0]
    
    # 5. Interpret the result
    # In your `create_generators`, Keras sorts class names alphabetically.
    # So: 'fake' = 0, 'real' = 1
    
    if prediction_prob > 0.5:
        label = 'REAL'
        confidence = prediction_prob * 100
    else:
        label = 'FAKE'
        confidence = (1 - prediction_prob) * 100
    
    print("\n--- PREDICTION RESULT ---")
    print(f"Model prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"(Raw score: {prediction_prob:.4f} -> 0=Fake, 1=Real)")


if __name__ == "__main__":
    # This allows you to pass the image path as a command-line argument
    
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]
        main(IMAGE_PATH)
    else:
        print("--- Error: No image path provided. ---")
        print("Please run this script from the command line like this:")
        print("python src/predict.py /path/to/your/image.jpg")