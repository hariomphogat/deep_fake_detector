import os
import sys
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import AUC
import matplotlib.pyplot as plt


# --- BEGIN ROBUST GPU FIX ---
# We must do this before any other TensorFlow operations
import tensorflow as tf
print("Applying robust GPU configuration...")
try:
    # Get all GPUs that TensorFlow can see
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            # Set memory growth to True for each GPU
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  > Enabled memory growth for: {gpu.name}")
    else:
        print("  > No GPUs found by TensorFlow. Will run on CPU.")
except Exception as e:
    print(f"  > Error applying GPU configuration: {e}")
# --- END ROBUST GPU FIX ---


# 1. --- Import from our project files ---
try:
    import config
    from model import build_baseline_model
except ImportError:
    print("Error: Could not import config.py or model.py.")
    print("Make sure they are in the 'src/' directory.")
    sys.exit(1)
    
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def create_balanced_dataframe():
    """
    Scans the training directory and creates a balanced DataFrame
    of 'real' and 'fake' image paths. This is our undersampling step.
    """
    print("Creating balanced training dataframe...")
    
    # Get lists of all real and fake training images
    real_paths = [os.path.join(config.TRAIN_REAL_DIR, f) for f in os.listdir(config.TRAIN_REAL_DIR) if f.endswith('.jpg')]
    fake_paths = [os.path.join(config.TRAIN_FAKE_DIR, f) for f in os.listdir(config.TRAIN_FAKE_DIR) if f.endswith('.jpg')]
    
    # Create DataFrames
    df_real = pd.DataFrame({'filepath': real_paths, 'label': 'real'})
    df_fake = pd.DataFrame({'filepath': fake_paths, 'label': 'fake'})
    
    # --- This is the key undersampling step ---
    # We sample the 'fake' DataFrame to have the same number of
    # images as the 'real' DataFrame.
    df_fake_sampled = df_fake.sample(n=len(df_real), random_state=42)
    
    # Combine and shuffle
    df_train_balanced = pd.concat([df_real, df_fake_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced training set created: {len(df_train_balanced)} total images")
    print(f"  Real: {len(df_real)} images")
    print(f"  Fake: {len(df_fake_sampled)} images")
    
    return df_train_balanced

def create_generators(train_df):
    """
    Creates the Keras Data Generators for training and validation.
    """
    print("Creating Data Generators...")
    
    # --- Training Generator with Data Augmentation ---
    # Data augmentation creates "new" versions of our images on-the-fly
    # (flipped, rotated, etc.) to make our model more robust.
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Normalize pixel values
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2     # We'll use 20% of our training data for validation
    )
    
    # --- Test/Validation Generator (No Augmentation) ---
    # We *never* augment our validation or test data.
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # --- Create the generators from our DataFrames ---
    
    # Training Generator (from the balanced DataFrame)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=(config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    # Validation Generator (also from the balanced DataFrame)
    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=(config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False  # No need to shuffle validation data
    )
    
    # Test Generator (from the *unbalanced* test directory)
    # This is our real-world test.
    test_generator = test_datagen.flow_from_directory(
        directory=config.TEST_DIR,
        target_size=(config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def plot_history(history, save_path):
    """
    Plots the training history (accuracy and loss) and saves it to a file.
    """
    print(f"Saving training history plot to {save_path}...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot training & validation accuracy values
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print("History plot saved.")


def main():
    """
    Main training function.
    """
    print("--- Phase 2: Starting Baseline Model Training ---")
    
    # 1. Handle class imbalance
    train_df = create_balanced_dataframe()
    
    # 2. Create data generators
    train_gen, val_gen, test_gen = create_generators(train_df)
    
    # 3. Build the model
    print("Building model...")
    model = build_baseline_model(config.TARGET_IMAGE_SIZE)
    
    # 4. Compile the model
    # We use AUC (Area Under the Curve) as our main metric.
    # It's much better than accuracy for imbalanced test sets.
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    
    model.summary()
    
    # 5. Define Callbacks
    # This will save the *best* model based on validation AUC
    checkpoint_path = os.path.join(config.MODEL_DIR, "baseline_model.h5")
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_auc',
        mode='max',
        verbose=1
    )
    
    # This will stop training early if it stops improving
    early_stopping = EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=5,  # Stop after 5 epochs of no improvement
        verbose=1,
        restore_best_weights=True
    )
    
    # 6. Start Training
    print("Starting model training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.n // config.BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.n // config.BATCH_SIZE,
        epochs=config.EPOCHS,
        callbacks=[model_checkpoint, early_stopping]
    )
    
    print("Training complete.")
    
    # 7. Evaluate on the (imbalanced) Test Set
    print("Evaluating model on the test set...")
    results = model.evaluate(test_gen, steps=test_gen.n // config.BATCH_SIZE)
    
    print("\n--- Test Set Evaluation ---")
    print(f"Test Loss:     {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test AUC:      {results[2]:.4f}")
    
    # 8. Save history plot
    plot_path = os.path.join(config.RESULTS_DIR, "baseline_training_history.png")
    plot_history(history, plot_path)
    
    print("\n--- Baseline Model Training Finished ---")
    print(f"Best model saved to: {checkpoint_path}")

if __name__ == "__main__":
    
    # Create models/ and results/ directories if they don't exist
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    main()