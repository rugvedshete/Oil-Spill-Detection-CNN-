# 🛢️ Oil Spill Detection — CNN (2025)

A **CNN-based Deep Learning project** that detects **oil spills in satellite imagery** to support faster and more accurate environmental monitoring.

✅ Built using **TensorFlow/Keras**  
✅ Includes a **trained model (`model.h5`)**  
✅ Web app deployed using **Flask** for real-time prediction  

---

## 🚀 Project Highlights

- Developed a **CNN model** to classify satellite images as:
  - **Oil Spill**
  - **No Oil Spill**
- Improved monitoring accuracy through model optimizations and preprocessing
- Created a simple **Flask web interface** for easy testing

---

## 🧰 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **CNN (Convolutional Neural Network)**
- **Flask**
- **NumPy, OpenCV, Matplotlib**

---

## 📁 Project Structure

```bash
Oil-Spill-Detection-main/
│── app.py                      # Flask web application
│── oil_spill.py                 # Main model/prediction logic
│── evaluate_model.py            # Model evaluation script
│── quick_accuracy_check.py      # Quick test script
│── model.h5                     # Trained CNN model
│── prediction_feedback.json     # Stores feedback/predictions
│── static/                      # Frontend static files (CSS/images)
│── templates/                   # UI pages (if present)
│── a.jpg, b.jpg, nooilspill.jpeg # Sample images
│── *.pdf / *.pptx / *.docx      # Reports & presentation files
