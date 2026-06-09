# 🚗 Driver Drowsiness Detection using Deep Learning

Real-time **Driver Drowsiness Detection System** built using **TensorFlow, OpenCV, and Streamlit**.

This application monitors eye state through webcam input and detects signs of drowsiness by classifying eyes as **Open** or **Closed**. If eyes remain closed for a configurable duration, the system triggers an alert.

# 📌 Overview

This project combines **Computer Vision** and **Deep Learning** to create a lightweight real-time monitoring system.

### Features

- Real-time webcam monitoring
- Face detection
- Geometric eye region extraction
- Deep Learning eye-state classification
- Adjustable drowsiness alert timer
- Temporal prediction smoothing
- Streamlit deployment

---

# 🧠 How It Works

```text
Webcam
↓
Face Detection
↓
Eye Region Extraction
↓
Preprocessing
↓
MobileNetV2 Prediction
↓
Temporal Smoothing
↓
Drowsiness Alert
```

---

# 📂 Dataset

The final dataset was created by combining multiple sources:

### Sources
- **MRL Eye Dataset**
- **CEW (Closed Eyes in the Wild) Dataset**
- **Custom eye images extracted using webcam**

### Final Dataset Structure


### Labels

| Class | Label |
|--------|-------|
| Closed | 0 |
| Open | 1 |

---

# 🏗️ Model

### Architecture

```text
Input (224×224)
↓
MobileNetV2
↓
GlobalAveragePooling
↓
Dense
↓
BatchNorm
↓
Dropout
↓
Sigmoid
↓
Open / Closed
```

### Why MobileNetV2?

- Lightweight
- Fast inference
- Transfer Learning
- Suitable for real-time applications

---

# 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | 92% |
| Precision | 92% |
| Recall | 92% |
| F1 Score | 92% |

---

# 📁 Project Structure

```text
Driver_Drowsiness_Detection/

│
├── app.py
├── training.ipynb
│
├── models/
│     └── drowsiness_modelretrained.keras
│
├── dataset/
│
├── extraction/
│     └── extract_eyes.py
│
├── haarcascades/
│     └── haarcascade_frontalface_default.xml
│
├── requirements.txt
│
└── README.md
```

---

# 👁️ Eye Extraction

Additional training data was created using a custom extraction pipeline.

Process:

```text
Image
↓
Face Detection
↓
Geometric Eye Extraction
↓
Combine Eyes
↓
Save Dataset
```


Add your facial images to extraction folder and run eye_extraction.ipynb to extract Eye regions.
Extracted images is stored in eye_dataset Folder

---

# ⚙️ Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

---

# 📦 Requirements

```text
tensorflow
opencv-python
streamlit
numpy
matplotlib
scikit-learn
pandas
```

---

# 🚀 Future Improvements

- MediaPipe face tracking
- Eye Aspect Ratio (EAR)
- Audio alert
- Low-light optimisation
- Mobile deployment

---

# 🛠 Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- NumPy

---

## 👨‍💻 Author

Developed as an academic Deep Learning project focused on real-time computer vision and deployment.
