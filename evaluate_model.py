from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_image(img_path):
    """Load and preprocess a single image"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def evaluate_model_accuracy(model_path, test_data_dir):
    """
    Evaluate model accuracy on test dataset
    
    Expected directory structure:
    test_data_dir/
    ├── oil_spill/     (images with oil spills)
    └── no_oil_spill/  (images without oil spills)
    """
    
    # Load the model
    model = load_model(model_path)
    
    # Prepare data
    true_labels = []
    predictions = []
    image_paths = []
    
    # Process oil spill images (label = 1)
    oil_spill_dir = os.path.join(test_data_dir, "oil_spill")
    if os.path.exists(oil_spill_dir):
        for filename in os.listdir(oil_spill_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(oil_spill_dir, filename)
                try:
                    img_array = load_and_preprocess_image(img_path)
                    prediction = model.predict(img_array, verbose=0)[0][0]
                    
                    true_labels.append(1)  # Oil spill
                    predictions.append(1 if prediction >= 0.5 else 0)
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    # Process no oil spill images (label = 0)
    no_oil_spill_dir = os.path.join(test_data_dir, "no_oil_spill")
    if os.path.exists(no_oil_spill_dir):
        for filename in os.listdir(no_oil_spill_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(no_oil_spill_dir, filename)
                try:
                    img_array = load_and_preprocess_image(img_path)
                    prediction = model.predict(img_array, verbose=0)[0][0]
                    
                    true_labels.append(0)  # No oil spill
                    predictions.append(1 if prediction >= 0.5 else 0)
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    if len(true_labels) == 0:
        print("No test images found. Please organize your test data in the expected structure.")
        return
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"\n=== Model Evaluation Results ===")
    print(f"Total test images: {len(true_labels)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    print(f"\n=== Classification Report ===")
    target_names = ['No Oil Spill', 'Oil Spill']
    print(classification_report(true_labels, predictions, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, true_labels, predictions

def evaluate_with_confidence_threshold(model_path, test_data_dir, confidence_threshold=0.7):
    """
    Evaluate model with different confidence thresholds
    Only make predictions when confidence is above threshold
    """
    model = load_model(model_path)
    
    true_labels = []
    predictions = []
    confidences = []
    uncertain_count = 0
    
    # Process all test images
    for class_name, label in [("oil_spill", 1), ("no_oil_spill", 0)]:
        class_dir = os.path.join(test_data_dir, class_name)
        if os.path.exists(class_dir):
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, filename)
                    try:
                        img_array = load_and_preprocess_image(img_path)
                        raw_prediction = model.predict(img_array, verbose=0)[0][0]
                        
                        # Calculate confidence
                        confidence = raw_prediction if raw_prediction >= 0.5 else 1 - raw_prediction
                        
                        if confidence >= confidence_threshold:
                            true_labels.append(label)
                            predictions.append(1 if raw_prediction >= 0.5 else 0)
                            confidences.append(confidence)
                        else:
                            uncertain_count += 1
                            
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
    
    if len(true_labels) > 0:
        accuracy = accuracy_score(true_labels, predictions)
        avg_confidence = np.mean(confidences)
        
        print(f"\n=== High-Confidence Predictions (threshold: {confidence_threshold}) ===")
        print(f"High-confidence predictions: {len(true_labels)}")
        print(f"Uncertain predictions (excluded): {uncertain_count}")
        print(f"Accuracy on high-confidence predictions: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Average confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    else:
        print(f"No predictions met the confidence threshold of {confidence_threshold}")

if __name__ == "__main__":
    # Example usage
    model_path = "model.h5"
    test_data_dir = "test_data"  # Create this directory with your test images
    
    print("=== Standard Evaluation ===")
    try:
        accuracy, true_labels, predictions = evaluate_model_accuracy(model_path, test_data_dir)
    except Exception as e:
        print(f"Error in standard evaluation: {e}")
        print("\nTo use this script, create a 'test_data' directory with subdirectories:")
        print("test_data/oil_spill/ (put oil spill images here)")
        print("test_data/no_oil_spill/ (put clean water images here)")
    
    print("\n" + "="*50)
    print("=== High-Confidence Evaluation ===")
    try:
        evaluate_with_confidence_threshold(model_path, test_data_dir, 0.8)
    except Exception as e:
        print(f"Error in confidence evaluation: {e}")