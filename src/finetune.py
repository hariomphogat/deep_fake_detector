import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import AUC
import matplotlib.pyplot as plt

# --- BEGIN ROBUST GPU FIX ---
# (Same as train.py)
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


# 1. --- Import from our project files ---
try:
    import config
    # We don't need build_baseline_model, we will load our saved one
except ImportError:
    print("Error: Could not import config.py.")
    sys.exit(1)
    
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# --- We can copy these functions directly from train.py ---
# --- (No changes needed) ---

def create_balanced_dataframe():
    """
    Scans the training directory and creates a balanced DataFrame
    of 'real' and 'fake' image paths. This is our undersampling step.
    """
    print("Creating balanced training dataframe...")
    real_paths = [os.path.join(config.TRAIN_REAL_DIR, f) for f in os.listdir(config.TRAIN_REAL_DIR) if f.endswith('.jpg')]
    fake_paths = [os.path.join(config.TRAIN_FAKE_DIR, f) for f in os.listdir(config.TRAIN_FAKE_DIR) if f.endswith('.jpg')]
    df_real = pd.DataFrame({'filepath': real_paths, 'label': 'real'})
    df_fake = pd.DataFrame({'filepath': fake_paths, 'label': 'fake'})
    df_fake_sampled = df_fake.sample(n=len(df_real), random_state=42)
    df_train_balanced = pd.concat([df_real, df_fake_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Balanced training set created: {len(df_train_balanced)} total images")
    return df_train_balanced

def create_generators(train_df):
    """
    Creates the Keras Data Generators for training and validation.
    """
    print("Creating Data Generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 
    )
    
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
    
    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=(config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False 
    )
    
    return train_generator, validation_generator

def plot_history(history, save_path):
    """
    Plots the training history (accuracy and loss) and saves it to a file.
    """
    print(f"Saving training history plot to {save_path}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    # ... (code is identical to train.py) ...
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    print("History plot saved.")

# --- END of copied functions ---


def main():
    """
    Main fine-tuning function.
    """
    print("--- Phase 2: Starting Model Fine-Tuning ---")
    
    # 1. We still need the generators for training
    train_df = create_balanced_dataframe()
    train_gen, val_gen = create_generators(train_df)
    
    # 2. --- NEW: Load the saved Phase 1 model ---
    print("Loading baseline model from Phase 1...")
    baseline_model_path = os.path.join(config.MODEL_DIR, "baseline_model.h5")
    if not os.path.exists(baseline_model_path):
        print(f"Error: Model file not found at {baseline_model_path}")
        print("Please run train.py first to create the baseline model.")
        return
        
    model = tf.keras.models.load_model(baseline_model_path)
    print("Baseline model loaded successfully.")

    # 3. --- NEW: Unfreeze the base model layers ---
    # The layers of the Xception model are *directly* in our loaded model.
    # We will unfreeze all layers from 'block10_sepconv1_act' onwards.
    # We MUST keep Batch Normalization layers frozen.

    UNFREEZE_FROM_LAYER = 'block10_sepconv1_act' # Unfreezes the last 5 blocks

    print(f"Unfreezing model from layer: {UNFREEZE_FROM_LAYER}")

    set_trainable = False
    for layer in model.layers:
        if layer.name == UNFREEZE_FROM_LAYER:
            set_trainable = True

        if set_trainable:
            # We are in the fine-tuning layers
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                # Unfreeze the layer
                layer.trainable = True
                # print(f"Unfreezing: {layer.name}") # Optional: Uncomment for debugging
            else:
                # IMPORTANT: Keep Batch Norm layers frozen
                layer.trainable = False
        else:
            # We are in the frozen layers
            layer.trainable = False

    print("Model layers un-frozen for fine-tuning.")

    # 4. --- NEW: Re-compile the model with a VERY low learning rate ---
    # This is the most important step of fine-tuning.
    # We use a "slow" optimizer so we don't destroy the weights we learned.
    
    # Your original LR was 2e-3 (0.002)
    # A good fine-tuning LR is 10-100x smaller.
    FINE_TUNE_LR = 2e-5  # (0.00002)
    
    print(f"Re-compiling model with a new learning rate: {FINE_TUNE_LR}")
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    
    model.summary() # Will now show many more trainable parameters

    # 5. --- NEW: Define new Callbacks for Phase 2 ---
    # We want to save our new model with a *new name*
    finetuned_checkpoint_path = os.path.join(config.MODEL_DIR, "finetuned_model.h5")
    
    model_checkpoint = ModelCheckpoint(
        filepath=finetuned_checkpoint_path,
        save_best_only=True,
        monitor='val_auc', # Still our most important metric
        mode='max',
        verbose=1
    )
    
    # Early stopping is still a good idea
    early_stopping = EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=5, # Stop after 5 epochs of no improvement
        verbose=1,
        restore_best_weights=True
    )
    
    # 6. Start Fine-Tuning
    print("Starting model fine-tuning...")
    
    # We are starting a *new* training run, so epochs can be 1-20
    # Let's aim for 15, but EarlyStopping will probably catch it
    FINETUNE_EPOCHS = 15
    
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.n // config.BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.n // config.BATCH_SIZE,
        epochs=FINETUNE_EPOCHS,
        callbacks=[model_checkpoint, early_stopping]
    )
    
    print("Fine-tuning complete.")
    
    # 7. Save history plot
    plot_path = os.path.join(config.RESULTS_DIR, "finetuned_training_history.png")
    plot_history(history, plot_path)
    
    print("\n--- Model Fine-Tuning Finished ---")
    print(f"Best fine-tuned model saved to: {finetuned_checkpoint_path}")

if __name__ == "__main__":
    main()