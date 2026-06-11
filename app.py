import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI Age & Gender Detection",
    page_icon="🎯",
    layout="wide"
)

# ---------------------------
# LOAD MODELS
# ---------------------------
@st.cache_resource
def load_models():

    yolo_model = YOLO("yolov8n.pt")

    age_model = load_model("age_model.h5")

    gender_model = load_model("gender_model.h5")

    return yolo_model, age_model, gender_model


yolo_model, age_model, gender_model = load_models()

# Labels
age_labels = ['Child', 'Teen', 'Young', 'Adult', 'Old']
gender_labels = ['Male', 'Female']

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("⚙️ Control Center")

st.sidebar.success("YOLOv8 Loaded")

st.sidebar.success("Age CNN Loaded")

st.sidebar.success("Gender CNN Loaded")

confidence = st.sidebar.slider(
"Detection Confidence",
0.1,
1.0,
0.5
)

st.sidebar.markdown("---")

st.sidebar.info(
"NeuralVision AI is actively monitoring "
"the webcam feed."
)
# ---------------------------
# TITLE
# ---------------------------
st.title("🎯 AI-Powered Age and Gender Detection System")

st.markdown(
    "Real-time age and gender prediction using "
    "YOLOv8, TensorFlow, Keras and OpenCV."
)

# ---------------------------
# START BUTTON
# ---------------------------
run = st.checkbox("📷 Start Webcam Detection")

FRAME_WINDOW = st.image([])

# Metrics placeholders
col1, col2, col3 = st.columns(3)

person_metric = col1.empty()
male_metric = col2.empty()
female_metric = col3.empty()

# ---------------------------
# WEBCAM
# ---------------------------
if run:

    cap = cv2.VideoCapture(0)

    while run:

        success, frame = cap.read()

        if not success:
            st.error("Unable to access webcam.")
            break

        results = yolo_model(frame)

        person_count = 0
        male_count = 0
        female_count = 0

        for result in results:

            boxes = result.boxes

            for box in boxes:

                cls = int(box.cls[0])

                # Person class
                if cls == 0:

                    person_count += 1

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    person = frame[y1:y2, x1:x2]

                    if person.size == 0:
                        continue

                    try:

                        img = cv2.resize(
                            person,
                            (224, 224)
                        )

                        img = img.astype(
                            "float32"
                        ) / 255.0

                        img = img_to_array(img)

                        img = np.expand_dims(
                            img,
                            axis=0
                        )

                        # AGE
                        age_pred = age_model.predict(
                            img,
                            verbose=0
                        )

                        age_idx = np.argmax(
                            age_pred
                        )

                        age_label = age_labels[
                            age_idx
                        ]

                        age_conf = (
                            np.max(age_pred)
                            * 100
                        )

                        # GENDER
                        gender_pred = gender_model.predict(
                            img,
                            verbose=0
                        )

                        gender_idx = np.argmax(
                            gender_pred
                        )

                        gender_label = gender_labels[
                            gender_idx
                        ]

                        gender_conf = (
                            np.max(gender_pred)
                            * 100
                        )

                        if gender_label == "Male":
                            color = (255, 0, 0)
                            male_count += 1
                        else:
                            color = (255, 0, 255)
                            female_count += 1

                        label = (
                            f"{gender_label}"
                            f" ({gender_conf:.1f}%)"
                            f" | "
                            f"{age_label}"
                            f" ({age_conf:.1f}%)"
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2
                        )

                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )

                    except Exception:
                        continue

        # Update metrics
        person_metric.metric(
            "Detected Persons",
            person_count
        )

        male_metric.metric(
            "Male",
            male_count
        )

        female_metric.metric(
            "Female",
            female_count
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        FRAME_WINDOW.image(
            frame,
            channels="RGB",
            use_container_width=True
        )

    cap.release()

else:
    st.info(
        "Enable 'Start Webcam Detection' "
        "to begin live prediction."
    )