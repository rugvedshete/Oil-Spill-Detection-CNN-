from flask import Flask, request, render_template_string, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime

app = Flask(__name__)
model = load_model("model.h5")  

STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

# File to store prediction feedback for accuracy calculation
FEEDBACK_FILE = "prediction_feedback.json"

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Oil Spill Detector</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
  <style>
    body {
      background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
      background-attachment: fixed;
      font-family: 'Segoe UI', sans-serif;
      color: white;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }
    .glass-card {
      background: rgba(255, 255, 255, 0.1);
      border-radius: 20px;
      padding: 40px;
      box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.18);
      width: 100%;
      max-width: 600px;
      animation: fadeIn 1s ease-in-out;
    }
    h1 {
      font-size: 2.5rem;
      font-weight: 600;
      animation: floatIn 1s ease-out;
    }
    .form-control {
      background: rgba(255, 255, 255, 0.2);
      border: none;
      color: white;
    }
    .form-control::file-selector-button {
      background-color: #00b894;
      color: white;
      padding: 10px 16px;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.3s ease;
    }
    .form-control::file-selector-button:hover {
      background-color: #019875;
    }
    .btn-custom {
      background-color: #6c5ce7;
      border: none;
      padding: 12px 24px;
      border-radius: 12px;
      color: white;
      font-size: 1rem;
      transition: 0.3s ease;
    }
    .btn-custom:hover {
      background-color: #a29bfe;
      color: black;
    }
    img {
      margin-top: 25px;
      border-radius: 15px;
      box-shadow: 0 5px 20px rgba(0, 0, 0, 0.5);
      max-width: 100%;
    }
    .result-text {
      margin-top: 20px;
      font-size: 1.2rem;
      animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
      0% { opacity: 0; transform: translateY(10px); }
      100% { opacity: 1; transform: translateY(0); }
    }

    @keyframes floatIn {
      0% { opacity: 0; transform: translateY(-20px); }
      100% { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="glass-card text-center">
    <h1><i class="fa-solid fa-water"></i> Oil Spill Detector</h1>
    <form method="POST" enctype="multipart/form-data" class="mt-4">
      <input class="form-control mb-3" type="file" name="image" required>
      <button type="submit" class="btn btn-custom">
        <i class="fa-solid fa-magnifying-glass"></i> Detect
      </button>
    </form>

    {% if result %}
      <div class="result-text">
        <strong>Prediction:</strong> {{ result.label }}<br>
        <strong>Confidence:</strong> {{ result.confidence }}
      </div>
      <img src="{{ url_for('static', filename=result.filename) }}" alt="Uploaded Image">
      
      <!-- Feedback buttons for accuracy tracking -->
      <div class="mt-3">
        <p><strong>Was this prediction correct?</strong></p>
        <button class="btn btn-success me-2" onclick="submitFeedback('{{ result.prediction_id }}', true)">
          <i class="fa-solid fa-check"></i> Correct
        </button>
        <button class="btn btn-danger" onclick="submitFeedback('{{ result.prediction_id }}', false)">
          <i class="fa-solid fa-times"></i> Incorrect
        </button>
      </div>
      <div id="feedback-message" class="mt-2"></div>
    {% endif %}
    
    <!-- Accuracy Statistics Button -->
    <div class="mt-4">
      <button class="btn btn-info" onclick="showAccuracy()">
        <i class="fa-solid fa-chart-bar"></i> View Model Accuracy
      </button>
    </div>
    <div id="accuracy-stats" class="mt-3"></div>
  </div>
  
  <script>
    function submitFeedback(predictionId, isCorrect) {
      fetch('/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prediction_id: predictionId,
          is_correct: isCorrect
        })
      })
      .then(response => response.json())
      .then(data => {
        document.getElementById('feedback-message').innerHTML = 
          '<div class="alert alert-success">Thank you for your feedback!</div>';
      })
      .catch(error => {
        document.getElementById('feedback-message').innerHTML = 
          '<div class="alert alert-danger">Error submitting feedback</div>';
      });
    }
    
    function showAccuracy() {
      fetch('/accuracy')
      .then(response => response.json())
      .then(data => {
        let html = '<div class="alert alert-info">';
        html += '<h5>Model Accuracy Statistics</h5>';
        html += `<p><strong>Total Predictions:</strong> ${data.total_predictions}</p>`;
        html += `<p><strong>Feedback Received:</strong> ${data.feedback_count}</p>`;
        if (data.feedback_count > 0) {
          html += `<p><strong>Accuracy:</strong> ${data.accuracy}% (${data.correct_predictions}/${data.feedback_count})</p>`;
        } else {
          html += '<p>No feedback received yet. Please provide feedback on predictions to calculate accuracy.</p>';
        }
        html += '</div>';
        document.getElementById('accuracy-stats').innerHTML = html;
      })
      .catch(error => {
        document.getElementById('accuracy-stats').innerHTML = 
          '<div class="alert alert-danger">Error loading accuracy stats</div>';
      });
    }
  </script>
</body>
</html>
'''

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)[0][0]
    label = "Oil Spill" if prediction >= 0.5 else "No Oil Spill"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    return label, float(confidence)

def save_prediction(prediction_id, label, confidence, filename):
    """Save prediction data for accuracy tracking"""
    prediction_data = {
        "id": prediction_id,
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "confidence": confidence,
        "filename": filename,
        "feedback": None
    }
    
    # Load existing predictions
    predictions = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                predictions = json.load(f)
        except:
            predictions = []
    
    predictions.append(prediction_data)
    
    # Save updated predictions
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(predictions, f, indent=2)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = file.filename
            filepath = os.path.join(STATIC_FOLDER, filename)
            file.save(filepath)
            label, confidence = predict_image(filepath)
            
            # Generate unique prediction ID
            prediction_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            
            result = {
                "label": label,
                "confidence": f"{confidence * 100:.2f}%",
                "filename": filename,
                "prediction_id": prediction_id
            }
            
            # Save prediction for accuracy tracking
            save_prediction(prediction_id, label, confidence, filename)
            
    return render_template_string(HTML_TEMPLATE, result=result)

@app.route("/feedback", methods=["POST"])
def feedback():
    """Handle user feedback on predictions"""
    data = request.get_json()
    prediction_id = data.get("prediction_id")
    is_correct = data.get("is_correct")
    
    # Load existing predictions
    predictions = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                predictions = json.load(f)
        except:
            return jsonify({"error": "Could not load feedback file"}), 500
    
    # Find and update the prediction
    for prediction in predictions:
        if prediction["id"] == prediction_id:
            prediction["feedback"] = is_correct
            break
    
    # Save updated predictions
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    return jsonify({"status": "success"})

@app.route("/accuracy")
def accuracy():
    """Get accuracy statistics"""
    predictions = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                predictions = json.load(f)
        except:
            predictions = []
    
    total_predictions = len(predictions)
    feedback_count = sum(1 for p in predictions if p["feedback"] is not None)
    correct_predictions = sum(1 for p in predictions if p["feedback"] is True)
    
    accuracy_percentage = 0
    if feedback_count > 0:
        accuracy_percentage = round((correct_predictions / feedback_count) * 100, 2)
    
    return jsonify({
        "total_predictions": total_predictions,
        "feedback_count": feedback_count,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy_percentage
    })

if __name__ == "__main__":
    app.run(debug=True)
