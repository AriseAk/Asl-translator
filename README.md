ASL-Translator 🤟
An end-to-end, real-time American Sign Language (ASL) alphabet translator. This project utilizes a deep learning pipeline to capture, detect, and classify hand gestures from a live webcam feed, translating them into text instantly.

🚀 Overview
The system is built as a distributed application with a React frontend and a Flask-PyTorch backend. Unlike traditional "shotgun" frame processing, this application uses a Smart Loop architecture to ensure smooth UI performance and high-quality predictions without overwhelming the server.

Technical Stack
Frontend: React, Vite, TailwindCSS, OGL (for the Starfield Background).

Backend: Python, Flask, PyTorch.

Models: ResNet50 (Classification), MediaPipe (Hand Localization).

🧠 The AI Pipeline
The translation process happens in three distinct stages:

Hand Localization: The backend uses MediaPipe to detect the presence of a hand in the raw frame. If found, it crops the image to the hand's bounding box. This drastically reduces "visual noise" from the background, allowing the classifier to focus purely on finger positions.

Classification: The cropped hand image is passed through a custom-trained ResNet50 neural network. The model was trained using stratified datasets and features a dropout-heavy architecture to improve generalization.

Stabilization: To prevent flickering, the backend maintains a Prediction Stabilizer. It uses a temporal buffer to ensure that a letter is only "confirmed" once it has been consistently predicted across multiple consecutive frames.

🛠️ Key Engineering Features
Smart Loop Request Locking
Instead of using a standard interval that fires frames blindly, the frontend implements a request-response lock. A new frame is only captured and sent once the backend has finished processing the previous one. This prevents "race conditions" where older predictions might overwrite newer ones, resulting in a stable, flicker-free user experience.

Dynamic Metadata Mapping
The model is truly self-contained. The .pth weight file includes a metadata dictionary of class names. When the backend starts, it automatically maps its output indices to the correct labels based on the file itself, removing the need for hardcoded class lists and preventing index-mismatch errors.

📂 Project Structure
Plaintext
├── backend/
│   ├── main.py            # Flask API & AI Logic
│   ├── asl_new.ipynb      # Training Notebook
│   └── asl_new_resnet50.pth   # Trained Model + Metadata
└── frontend/
    ├── src/
    │   ├── App.jsx        # Smart-Loop UI Logic
    │   ├── App.css        # Sage/Beige Styled UI
    │   └── components/    # Galaxy Background Component
🚥 Getting Started
Prerequisites
Python 3.11+

Node.js & npm

A CUDA-enabled GPU (optional, but recommended for real-time performance)

Installation
Backend Setup:

Bash
cd backend
pip install torch torchvision mediapipe flask flask-cors opencv-python pillow
python main.py
Frontend Setup:

Bash
cd frontend
npm install
npm run dev
📝 Observations
While the model shows exceptional performance on standardized datasets, real-world performance is highly dependent on lighting and background neutrality. The system performs best when the hand is well-lit and the background is uncluttered.
