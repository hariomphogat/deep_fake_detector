import os
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def build_baseline_model(image_size):
    """
    Builds a baseline CNN model using Xception as a base.
    """
    
    # 1. Define the input shape
    input_tensor = Input(shape=(image_size, image_size, 3))
    
    # 2. Load the Xception base model, pre-trained on ImageNet.
    # We don't include the final classification layer (include_top=False).
    base_model = Xception(
        weights='imagenet', 
        include_top=False, 
        input_tensor=input_tensor
    )
    
    # 3. Freeze the base model's layers
    # We do this so we only train our new "head" layers
    base_model.trainable = False
    
    # 4. Add our custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Condenses the features
    x = Dropout(0.5)(x)             # Adds regularization to prevent overfitting
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # 5. Add the final output layer
    # Sigmoid activation is used for binary (0 or 1) classification
    output_tensor = Dense(1, activation='sigmoid', name='output')(x)
    
    # 6. Create the final model
    model = Model(inputs=input_tensor, outputs=output_tensor)
    
    return model

if __name__ == "__main__":
    # A quick test to see if the model builds correctly
    print("Building test model...")
    # Import image size from our config
    try:
        from config import TARGET_IMAGE_SIZE
        model = build_baseline_model(TARGET_IMAGE_SIZE)
        model.summary()
        print("Model built successfully!")
    except ImportError:
        print("Error: Could not import TARGET_IMAGE_SIZE from config.")
    except Exception as e:
        print(f"Error building model: {e}")