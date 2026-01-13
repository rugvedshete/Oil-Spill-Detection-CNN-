import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

def quick_accuracy_check(model_path, images_dir, sample_size=50):
    """
    Quick accuracy check using existing images
    Manually label a sample of images and check accuracy
    """
    
    model = load_model(model_path)
    
    # Get random sample of images
    all_images = []
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_images.append(filename)
    
    if len(all_images) < sample_size:
        sample_size = len(all_images)
    
    sample_images = random.sample(all_images, sample_size)
    
    print(f"=== Manual Accuracy Check ===")
    print(f"Sample size: {sample_size} images")
    print("\nFor each image, you'll see the model's prediction.")
    print("Please enter 'y' if correct, 'n' if incorrect, 's' to skip:")
    print("-" * 50)
    
    correct_predictions = 0
    total_evaluated = 0
    
    for i, filename in enumerate(sample_images, 1):
        img_path = os.path.join(images_dir, filename)
        
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Make prediction
            prediction = model.predict(img_array, verbose=0)[0][0]
            label = "Oil Spill" if prediction >= 0.5 else "No Oil Spill"
            confidence = prediction if prediction >= 0.5 else 1 - prediction
            
            print(f"\n[{i}/{sample_size}] Image: {filename}")
            print(f"Prediction: {label} (Confidence: {confidence*100:.1f}%)")
            
            # Get user feedback
            while True:
                user_input = input("Is this correct? (y/n/s): ").lower().strip()
                if user_input in ['y', 'yes']:
                    correct_predictions += 1
                    total_evaluated += 1
                    break
                elif user_input in ['n', 'no']:
                    total_evaluated += 1
                    break
                elif user_input in ['s', 'skip']:
                    break
                else:
                    print("Please enter 'y' for yes, 'n' for no, or 's' to skip")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if total_evaluated > 0:
        accuracy = correct_predictions / total_evaluated
        print(f"\n=== Results ===")
        print(f"Images evaluated: {total_evaluated}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Estimated accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    else:
        print("No images were evaluated.")

if __name__ == "__main__":
    model_path = "model.h5"
    images_dir = "static"  # Using your existing static folder
    
    quick_accuracy_check(model_path, images_dir, sample_size=20)