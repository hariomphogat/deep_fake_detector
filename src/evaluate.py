import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. --- Import config ---
try:
    import config
except ImportError:
    print("Error: Could not import config.py.")
    sys.exit(1)

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def create_test_generator():
    """
    Creates the Keras Data Generator for the test set.
    """
    print("Creating Test Data Generator...")
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        directory=config.TEST_DIR,
        target_size=(config.TARGET_IMAGE_SIZE, config.TARGET_IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False  # IMPORTANT: Must be False for evaluation
    )
    
    return test_generator

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

def main():
    print("--- Starting Final Model Evaluation ---")
    
    # 1. Load our test data
    test_gen = create_test_generator()
    
    # 2. Load our BEST model (the fine-tuned one)
    model_path = os.path.join(config.MODEL_DIR, "finetuned_model.h5")
    if not os.path.exists(model_path):
        print(f"Error: Fine-tuned model not found at {model_path}")
        print("Run finetune.py first.")
        return
        
    print(f"Loading final model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 3. Get model's performance on the test set
    print("Evaluating model on test set...")
    results = model.evaluate(test_gen, verbose=1)
    
    print("\n--- Test Set Evaluation Metrics ---")
    print(f"Test Loss:     {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test AUC:      {results[2]:.4f}")
    
    # 4. Get predictions for the entire test set
    print("Generating predictions for classification report...")
    # This gets the raw probabilities (e.g., 0.1, 0.8, 0.9)
    y_pred_probs = model.predict(test_gen, verbose=1)
    # We convert probabilities to class labels (0 or 1)
    y_pred_classes = (y_pred_probs > 0.5).astype(int)
    
    # 5. Get the true labels
    y_true = test_gen.classes
    
    # 6. Get the class names
    class_names = list(test_gen.class_indices.keys())
    
    # 7. Print Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # 8. Save Confusion Matrix
    cm_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred_classes, cm_path)

if __name__ == "__main__":
    # Create the results directory if it doesn't exist
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    main()