import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# We still need config for the image size, so this import is fine
try:
    from . import config
except ImportError:
    import config

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def get_image_prediction(image_path, model):
    
    # 1. Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # 2. Make a prediction (using the model passed as an argument)
    prediction_prob = model.predict(processed_image)[0][0]
    
    # 3. Interpret the result
    # 'fake' = 0, 'real' = 1 (based on our generator)
    if prediction_prob > 0.5:
        label = 'REAL'
        confidence = prediction_prob * 100
    else:
        label = 'FAKE'
        confidence = (1 - prediction_prob) * 100
    
    # 4. Return the results in a dictionary
    return {
        "prediction": label,
        "confidence": float(confidence),
        "raw_score": float(prediction_prob)
    }
