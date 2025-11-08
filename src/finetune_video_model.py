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
    # We still need the generator to feed the model
    from video_data_generator import VideoDataGenerator
    # We don't need the builder, we're loading a saved model
except ImportError:
    print("Error: Could not import config.py or video_data_generator.py.")
    sys.exit(1)

# --- 2. GPU Fix ---
print("Applying robust GPU configuration...")
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  > Enabled memory growth for: {gpu.name}")
except Exception as e:
    print(f"  > Error applying GPU configuration: {e}")

# --- 3. Plot History Function (Identical to train_video_model.py) ---
def plot_history(history, save_path):
    print(f"Saving training history plot to {save_path}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    # Accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    # Loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    print("History plot saved.")

# --- 4. Main Fine-Tuning Function ---
def main():
    """
    Main fine-tuning function for the CNN-LSTM video model.
    """
    print("--- Phase 2: Starting CNN-LSTM Model Fine-Tuning ---")
    
    # 1. Instantiate Data Generators (Same as before)
    print("Initializing data generators...")
    train_gen = VideoDataGenerator(
        data_dir=config.TRAIN_SEQ_DIR,
        batch_size=config.VIDEO_BATCH_SIZE, # Still use the small batch size
        sequence_length=config.SEQUENCE_LENGTH,
        img_size=config.TARGET_IMAGE_SIZE,
        shuffle=True
    )
    val_gen = VideoDataGenerator(
        data_dir=config.TEST_SEQ_DIR,
        batch_size=config.VIDEO_BATCH_SIZE,
        sequence_length=config.SEQUENCE_LENGTH,
        img_size=config.TARGET_IMAGE_SIZE,
        shuffle=False
    )
    
    # 2. --- NEW: Load the Phase 1 Model ---
    model_path = os.path.join(config.MODEL_DIR, "cnn_lstm_video_model.h5")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run train_video_model.py first.")
        return
        
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 3. --- NEW: Unfreeze the Encoder ---
    # We need to unfreeze the 'finetuned_encoder' (the Xception part)
    # that we wrapped inside the 'TimeDistributed' layer.
    
    print("Unfreezing the encoder (finetuned_encoder) for fine-tuning...")
    
    # Find the TimeDistributed layer by its name
    # The default name is 'time_distributed' if we didn't name it
    # We should find its name from the model summary
    
    # Let's find the encoder layer by its name "finetuned_encoder"
    # This is the *safer* way
    
    # Find the 'TimeDistributed' wrapper
    time_dist_layer = model.get_layer('time_distributed')
    
    # Get the 'base_model' (our encoder) from *inside* the wrapper
    encoder = time_dist_layer.layer
    
    # Set the encoder to be trainable
    encoder.trainable = True
    
    # Optional: Only unfreeze the *top* blocks, just like our image model
    # This is safer and highly recommended.
    
    # Let's find the layer to unfreeze from
    unfreeze_layer_name = 'block10_sepconv1_act'
    set_trainable = False
    
    print(f"Unfreezing from layer: {unfreeze_layer_name}")
    for layer in encoder.layers:
        if layer.name == unfreeze_layer_name:
            set_trainable = True
            
        if set_trainable:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True # Unfreeze
            else:
                layer.trainable = False # Keep BN frozen
        else:
            layer.trainable = False # Keep frozen
            
    print("Encoder layers un-frozen for fine-tuning.")

    # 4. --- NEW: Re-compile with a VERY low learning rate ---
    # This is the most important step.
    FINE_TUNE_LR = 2e-5  # (0.00002) - 100x smaller than before
    
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR), # Use the new LR
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    
    print(f"Model re-compiled with new learning rate: {FINE_TUNE_LR}")
    model.summary() # You will see many more "Trainable params"
    
    # 5. --- NEW: Define Callbacks for Fine-Tuning ---
    # Save the final model with a new name
    finetuned_path = os.path.join(config.MODEL_DIR, "finetuned_video_model.h5")
    
    model_checkpoint = ModelCheckpoint(
        filepath=finetuned_path,
        save_best_only=True,
        monitor='val_auc', # Still our best metric
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=5,  # Be less patient now
        verbose=1,
        restore_best_weights=True
    )
    
    # 6. Start Fine-Tuning
    print("Starting model FINE-TUNING...")
    
    # We are starting a *new* training run
    FINETUNE_EPOCHS = 20 # Let's aim for 20, EarlyStopping will catch it
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINETUNE_EPOCHS,
        callbacks=[model_checkpoint, early_stopping]
    )
    
    print("Fine-tuning complete.")
    
    # 7. Save history plot with a new name
    plot_path = os.path.join(config.RESULTS_DIR, "finetuned_video_history.png")
    plot_history(history, plot_path)
    
    print("\n--- CNN-LSTM Model Fine-Tuning Finished ---")
    print(f"Best fine-tuned video model saved to: {finetuned_path}")

# --- 5. Run the Script ---
if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    main()