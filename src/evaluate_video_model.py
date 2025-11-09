import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. --- Import config and generator ---
try:
    import config
    from video_data_generator import VideoDataGenerator
except ImportError:
    print("Error: Could not import config.py or video_data_generator.py.")
    sys.exit(1)

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'], # Swapped for correct labels
                yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix (Video Model)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

def main():
    print("--- Starting Video Model Evaluation ---")
    
    # 1. Load our test data using the VideoDataGenerator
    print("Creating Test Data Generator...")
    test_gen = VideoDataGenerator(
        data_dir=config.TEST_SEQ_DIR,
        batch_size=config.VIDEO_BATCH_SIZE, # Use the video batch size
        sequence_length=config.SEQUENCE_LENGTH,
        img_size=config.TARGET_IMAGE_SIZE,
        shuffle=False # IMPORTANT: Must be False for evaluation
    )
    
    # 2. Load our BEST fine-tuned video model
    model_path = os.path.join(config.MODEL_DIR, "finetuned_video_model.h5")
    if not os.path.exists(model_path):
        print(f"Error: Fine-tuned model not found at {model_path}")
        return
        
    print(f"Loading final model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 3. Get model's performance on the test set
    print("Evaluating model on test set...")
    # 'test_gen.n' isn't defined, so we just use the generator directly
    results = model.evaluate(test_gen, verbose=1)
    
    print("\n--- Test Set Evaluation Metrics ---")
    print(f"Test Loss:     {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test AUC:      {results[2]:.4f}")
    
    # 4. Get predictions for the entire test set
    print("Generating predictions for classification report...")
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred_classes = (y_pred_probs > 0.5).astype(int)
    
    # 5. Get the true labels
    # We get them from the generator's dataframe
    y_true = test_gen.df['label'].map({'fake': 0, 'real': 1}).values
    
    # We must slice y_true to match the number of predictions
    # This handles any batches that were dropped
    y_true = y_true[:len(y_pred_classes)]
    
    # 6. Print Classification Report
    print("\n--- Classification Report (Video Model) ---")
    print(classification_report(y_true, y_pred_classes, target_names=['fake', 'real']))
    
    # 7. Save Confusion Matrix
    cm_path = os.path.join(config.RESULTS_DIR, "video_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred_classes, cm_path)

if __name__ == "__main__":
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    main()