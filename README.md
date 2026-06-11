# 🤖 NeuralVision AI: Real-Time Age & Gender Analytics using YOLOv8

An AI-powered computer vision system that detects people in real-time using YOLOv8 and predicts their age group and gender using custom-trained deep learning models.

---

## 🚀 Features

✅ Real-time webcam-based person detection

✅ YOLOv8-powered object detection

✅ Custom-trained CNN model for age prediction

✅ Custom-trained CNN model for gender prediction

✅ Live age and gender classification

✅ Interactive Streamlit dashboard

✅ Detection statistics and analytics

✅ Confidence score display

✅ Modern AI-inspired user interface

---

## 🛠 Technologies Used

- Python
- YOLOv8
- OpenCV
- TensorFlow
- Keras
- NumPy
- Streamlit
- Deep Learning
- Computer Vision

---

## 📂 Project Structure

```text
AI-Powered-Age-and-Gender-Detection-System/
│
├── app.py
├── train_age_model.py
├── train_gender_model.py
├── age_model.h5
├── gender_model.h5
├── yolov8n.pt
├── requirements.txt
├── README.md
├── .gitignore
├── screenshots/
└── sample_images/
```

---

## 🧠 How It Works

1. Webcam captures live video frames.
2. YOLOv8 detects people in the frame.
3. Detected regions are extracted.
4. Images are preprocessed and resized.
5. Age prediction model predicts age group.
6. Gender prediction model predicts gender.
7. Results are displayed with bounding boxes and confidence scores.

---

## 📊 Age Categories

The model classifies people into the following age groups:

- Child
- Teen
- Young
- Adult
- Old

---

## 👤 Gender Categories

- Male
- Female

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/adhithyanms25/AI-Powered-Age-and-Gender-Detection-System.git
cd AI-Powered-Age-and-Gender-Detection-System
```

### 2. Create Virtual Environment

```bash
python -m venv myenv
```

### 3. Activate Environment

Windows:

```bash
myenv\Scripts\activate
```

Linux/Mac:

```bash
source myenv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 📸 Screenshots

### Dashboard

<img width="960" height="504" alt="home" src="https://github.com/user-attachments/assets/9aa7c2a4-0c27-4833-892b-bc17f840197a" />

### Live Detection

Add webcam detection screenshot here

```text
screenshots/live_detection.png
```

### Multiple Person Detection

Add detection results screenshot here

```text
screenshots/multiple_detection.png
```

---

## 📈 Future Enhancements

- Emotion Detection
- Face Recognition
- Attendance System Integration
- Age Estimation Refinement
- Cloud Deployment
- Mobile Application Support

---

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- Deep Learning Model Training
- Computer Vision Applications
- YOLOv8 Object Detection
- TensorFlow & Keras
- Real-Time AI Systems
- Streamlit Dashboard Development
- Model Deployment

---

## 👨‍💻 Author

### Adhithyan M S

BCA Graduate | AI & Data Science Enthusiast

GitHub:
https://github.com/adhithyanms25

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.
