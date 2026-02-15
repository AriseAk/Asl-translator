# ASL-Translator 🤟

**Real-Time American Sign Language (ASL) Alphabet Translator**

An end-to-end, real-time ASL alphabet translation system that captures hand gestures from a live webcam feed and converts them into text instantly using a deep learning inference pipeline.

---

## 🚀 Overview

ASL-Translator is designed as a distributed AI application with a high-performance inference backend and a responsive, visually immersive frontend.

Unlike traditional frame-sampling systems that send frames at fixed intervals (“shotgun processing”), this project implements a **Smart Loop Architecture** that synchronizes frontend capture with backend inference. This ensures:

* Smooth UI rendering
* Reduced server overload
* Higher prediction stability
* Elimination of race conditions

---

## 🧩 Technical Stack

**Frontend**

* React
* Vite
* TailwindCSS
* OGL (Starfield / Galaxy background rendering)

**Backend**

* Python
* Flask
* PyTorch

**Models & CV Tooling**

* ResNet50 → Gesture Classification
* MediaPipe → Hand Detection & Localization

---

## 🧠 AI Inference Pipeline

The translation pipeline operates in three sequential stages:

### 1️⃣ Hand Localization

* MediaPipe detects the presence of a hand within the incoming webcam frame.
* If detected, the system extracts the bounding box region.
* The frame is cropped to isolate the hand.

**Why this matters:**
Reducing background pixels removes visual noise and ensures the classifier focuses strictly on finger articulation and pose geometry.

---

### 2️⃣ Classification

* The cropped hand image is passed into a custom-trained **ResNet50** model.
* The network was trained on stratified ASL alphabet datasets.
* A dropout-heavy architecture was used to improve generalization and reduce overfitting.

**Output:**
A probability distribution across ASL alphabet classes.

---

### 3️⃣ Prediction Stabilization

Real-time classifiers often produce flickering predictions. To mitigate this:

* The backend maintains a **Temporal Prediction Buffer**.
* Predictions are accumulated across consecutive frames.
* A letter is only confirmed once it achieves consistency within the buffer window.

**Result:**
Stable, human-readable text output without rapid character switching.

---

## 🛠️ Key Engineering Features

### 🔁 Smart Loop Request Locking

Instead of using a fixed interval capture system:

* The frontend sends a frame **only after** the backend finishes processing the previous request.
* This creates a request-response lock cycle.

**Benefits:**

* Prevents race conditions
* Avoids stale predictions overwriting new ones
* Reduces unnecessary compute load
* Maintains UI smoothness

---

### 🧬 Dynamic Metadata Mapping

The trained model is self-describing.

* The `.pth` weight file embeds a metadata dictionary containing class labels.
* On backend initialization, label indices are auto-mapped from this metadata.

**Advantages:**

* No hardcoded class lists
* Eliminates index mismatch bugs
* Improves portability of trained models

---

## 📂 Project Structure

```
ASL-Translator/
│
├── backend/
│   ├── main.py                # Flask API + Inference Pipeline
│   ├── asl_new.ipynb          # Model Training Notebook
│   └── asl_new_resnet50.pth   # Trained Weights + Metadata
│
└── frontend/
    ├── src/
    │   ├── App.jsx            # Smart Loop Capture Logic
    │   ├── App.css            # Sage/Beige UI Theme
    │   └── components/        # Starfield Background (OGL)
```

---

## 🚥 Getting Started

### Prerequisites

* Python 3.11+
* Node.js + npm
* Webcam access
* CUDA-enabled GPU *(optional but recommended for real-time inference)*

---

## ⚙️ Installation & Setup

### Backend Setup

```bash
cd backend

pip install torch torchvision mediapipe flask flask-cors opencv-python pillow

python main.py
```

---

### Frontend Setup

```bash
cd frontend

npm install
npm run dev
```

The frontend will start on the default Vite development server (typically `localhost:5173`).

---

## 🧪 Performance Observations

* The model achieves high accuracy on standardized datasets.
* Real-world inference performance depends on environmental factors.

### Optimal Conditions

* Well-lit hand region
* Neutral / uncluttered background
* Minimal motion blur
* Clear finger articulation

### Limiting Factors

* Poor lighting
* Complex backgrounds
* Partial hand occlusion
* Extreme camera angles

---

## 📌 Future Improvements

* Word-level gesture modeling
* Sequence decoding (LSTM / Transformer integration)
* Multi-hand detection
* Mobile deployment (TensorFlow Lite / ONNX)
* Dataset expansion for robustness

---

**Built to bridge communication gaps using real-time AI.**
