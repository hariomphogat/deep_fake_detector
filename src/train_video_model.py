import os
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import AUC
import tensorflow as tf

# --- 1. Import our project files ---
try:
    import config
    # Import our new generator and model
    from video_data_generator import VideoDataGenerator
    from video_model import build_video_model
except ImportError:
    print("Error: Could not import config.py, video_data_generator.py, or video_model.py.")
    print("Make sure they are all in the 'src/' directory.")
    sys.exit(1)

# --- 2. BEGIN ROBUST GPU FIX ---
# (Same as your original train.py)
print("Applying robust GPU configuration...")
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  > Enabled memory growth for: {gpu.name}")
    else:
        print("  > No GPUs found by TensorFlow. Will run on CPU.")
except Exception as e:
    print(f"  > Error applying GPU configuration: {e}")
# --- END ROBUST GPU FIX ---

# --- 3. Plot History Function ---
# (Copied from train.py)
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

# --- 4. Main Training Function ---
def main():
    """
    Main training function for the new CNN-LSTM video model.
    """
    print("--- Phase 1: Starting CNN-LSTM Model Training ---")
    
    # 1. Instantiate Data Generators
    # We use our new VideoDataGenerator and the new paths
    print("Initializing data generators...")
    train_gen = VideoDataGenerator(
        data_dir=config.TRAIN_SEQ_DIR,
        batch_size=config.VIDEO_BATCH_SIZE, # Using the new, smaller batch size
        sequence_length=config.SEQUENCE_LENGTH,
        img_size=config.TARGET_IMAGE_SIZE,
        shuffle=True
    )
    
    val_gen = VideoDataGenerator(
        data_dir=config.TEST_SEQ_DIR, # Using the TEST set for validation
        batch_size=config.VIDEO_BATCH_SIZE,
        sequence_length=config.SEQUENCE_LENGTH,
        img_size=config.TARGET_IMAGE_SIZE,
        shuffle=False # No need to shuffle validation data
    )
    
    # 2. Build the model
    print("Building model...")
    model = build_video_model()
    
    # 3. Compile the model
    # We are training the new LSTM head
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')] # AUC is a great metric
    )
    
    model.summary()
    
    # 4. Define Callbacks
    # Save the new model with a new name
    checkpoint_path = os.path.join(config.MODEL_DIR, "cnn_lstm_video_model.h5")
    
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_auc', # Monitor our best metric
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=7,  # Stop after 7 epochs of no improvement
        verbose=1,
        restore_best_weights=True # Restore the best model
    )
    
    # 5. Start Training
    print("Starting model training...")
    
    # We set epochs high and let EarlyStopping find the best one
    NUM_EPOCHS = 50 
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=NUM_EPOCHS,
        callbacks=[model_checkpoint, early_stopping]
        # No 'steps_per_epoch' needed!
        # Keras automatically knows the length from our
        # generator's `__len__` method.
    )
    
    print("Training complete.")
    
    # 6. Save history plot with a new name
    plot_path = os.path.join(config.RESULTS_DIR, "cnn_lstm_training_history.png")
    plot_history(history, plot_path)
    
    print("\n--- CNN-LSTM Model Training Finished ---")
    print(f"Best video model saved to: {checkpoint_path}")

# --- 5. Run the Script ---
if __name__ == "__main__":
    
    # --- CRITICAL FIX ---
    # Create models/ and results/ directories if they don't exist
    # This prevents the FileNotFoundError
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    main()