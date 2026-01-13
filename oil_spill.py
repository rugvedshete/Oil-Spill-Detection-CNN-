from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf
import os

# Create FastAPI app instance
app = FastAPI()

# Define model path (make sure to replace this with the correct path to your model)
model_path = 'E:/mini-project/model.h5'

# Check if the model file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the model
model = tf.keras.models.load_model(model_path)

def preprocess_image(contents):
    # Open the image with PIL
    img = Image.open(BytesIO(contents))

    # Resize the image to (224, 224) as required by the model
    img = img.resize((224, 224))

    # Convert the image to a numpy array and normalize it
    img_array = np.array(img) / 255.0  # Normalize to range [0, 1]

    # Ensure the image has 3 channels (RGB)
    if img_array.shape[-1] != 3:
        raise ValueError("Image must have 3 color channels (RGB)")

    # Add batch dimension (since the model expects shape (None, 224, 224, 3))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file contents
        contents = await file.read()

        # Preprocess the image
        image = preprocess_image(contents)

        # Perform prediction
        prediction = model.predict(image)[0][0]

        # Determine the label based on prediction
        label = "Oil Spill" if prediction >= 0.5 else "Not Oil Spill"
        confidence = float(prediction)

        return {"label": label, "confidence": confidence}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
