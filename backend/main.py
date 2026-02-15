import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from collections import deque, Counter
import time
import cv2
import os

# ============================================
# SYSTEM SETTINGS
# ============================================
# Suppress TensorFlow/MediaPipe logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("⚠️  MediaPipe not installed. Run: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False

# ============================================
# CONFIGURATION
# ============================================
class Config:
    MODEL_PATH = 'asl_new_resnet50.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tuning Parameters
    CONFIDENCE_THRESHOLD = 0.60
    BUFFER_SIZE = 5
    CHANGE_THRESHOLD = 0.6
    MIN_SAME_PREDICTIONS = 3
    
    # Image Settings
    IMAGE_SIZE = 224
    USE_HAND_DETECTION = True 
    HAND_MARGIN = 40 

# ============================================
# HAND DETECTOR (MediaPipe)
# ============================================
class HandDetector:
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE or not Config.USE_HAND_DETECTION:
            self.enabled = False
            return
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True, # Better for independent frames
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.enabled = True
        print("✓ MediaPipe Hand Detection Active")
    
    def detect_and_crop(self, image):
        if not self.enabled: return image
        
        # Convert PIL to Numpy
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        # MediaPipe expects RGB
        img_rgb = img_array
        
        results = self.hands.process(img_rgb)
        
        if not results.multi_hand_landmarks:
            return image # Return original if no hand found
        
        # Get Bounding Box
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = img_rgb.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        # Add margin
        x_min = max(0, int(min(x_coords)) - Config.HAND_MARGIN)
        x_max = min(w, int(max(x_coords)) + Config.HAND_MARGIN)
        y_min = max(0, int(min(y_coords)) - Config.HAND_MARGIN)
        y_max = min(h, int(max(y_coords)) + Config.HAND_MARGIN)
        
        # Crop
        hand_crop = img_rgb[y_min:y_max, x_min:x_max]
        
        if hand_crop.size == 0: return image
        
        return Image.fromarray(hand_crop)

# ============================================
# DYNAMIC MODEL WRAPPER
# ============================================
class ASLModel:
    def __init__(self, model_path, device):
        self.device = device
        print(f"Loading model from: {model_path}")
        
        # 1. LOAD CHECKPOINT & METADATA
        try:
            checkpoint = torch.load(model_path, map_location=device)
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load .pth file. {e}")
            raise e

        # 2. EXTRACT CLASS NAMES
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
            print(f"✓ Found metadata! Mapped {len(self.class_names)} classes.")
        else:
            # Fallback if metadata is missing (should not happen with new training)
            print("⚠️ Metadata not found. Using default classes.")
            self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                                'del', 'nothing', 'space']
            
        self.idx_to_class = {i: name for i, name in enumerate(self.class_names)}
        
        # 3. BUILD MODEL ARCHITECTURE
        # Use weights=None to avoid the "pretrained" warning
        self.model = models.resnet50(weights=None)
        
        # Replace output layer to match the number of classes in the file
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(self.class_names))
        )
        
        # 4. LOAD WEIGHTS
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(device).eval()
        
        # 5. PREPROCESSING (Must match Training!)
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            # NORMALIZATION ENABLED (Matches your training code)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("✓ Model Loaded Successfully")

    @torch.no_grad()
    def predict(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor), dim=1)[0]
        confidence, idx = torch.max(probs, 0)
        
        all_probs = {self.idx_to_class[i]: p.item() for i, p in enumerate(probs)}
        
        return self.idx_to_class[idx.item()], confidence.item(), all_probs

# ============================================
# STABILIZER
# ============================================
class PredictionStabilizer:
    def __init__(self):
        self.buffer = deque(maxlen=Config.BUFFER_SIZE)
        self.confidences = deque(maxlen=Config.BUFFER_SIZE)
        self.current_letter = None
        self.current_confidence = 0.0
        self.stats = {'total': 0, 'accepted': 0, 'rejected': 0}

    def update(self, letter, confidence):
        self.stats['total'] += 1
        self.buffer.append(letter)
        self.confidences.append(confidence)

        if not self.current_letter:
            self.current_letter = letter
            self.current_confidence = confidence
            return True, letter, confidence

        counts = Counter(self.buffer)
        most_common, count = counts.most_common(1)[0]
        agreement = count / len(self.buffer)
        
        relevant_confs = [c for l, c in zip(self.buffer, self.confidences) if l == most_common]
        avg_conf = np.mean(relevant_confs) if relevant_confs else 0

        is_strong = (
            agreement >= Config.CHANGE_THRESHOLD and 
            count >= Config.MIN_SAME_PREDICTIONS and 
            avg_conf >= Config.CONFIDENCE_THRESHOLD
        )

        if is_strong and (most_common != self.current_letter or avg_conf > self.current_confidence):
            self.stats['accepted'] += 1
            self.current_letter = most_common
            self.current_confidence = avg_conf
            return True, most_common, avg_conf

        self.stats['rejected'] += 1
        return False, self.current_letter, self.current_confidence

    def reset(self):
        self.buffer.clear()
        self.confidences.clear()
        self.current_letter = None

# ============================================
# FLASK APP
# ============================================
app = Flask(__name__)
CORS(app)

# Global Variables
model_instance = None
hand_detector = None
stabilizer = None

def initialize_backend():
    global model_instance, hand_detector, stabilizer
    print("\n" + "="*50)
    print("INITIALIZING BACKEND...")
    try:
        model_instance = ASLModel(Config.MODEL_PATH, Config.DEVICE)
        hand_detector = HandDetector()
        stabilizer = PredictionStabilizer()
        print("✓ BACKEND READY")
    except Exception as e:
        print(f"❌ STARTUP FAILED: {e}")
        model_instance = None
    print("="*50 + "\n")

# Run initialization
initialize_backend()

@app.route('/predict', methods=['POST'])
def predict():
    if not model_instance: return jsonify({'error': 'Not loaded'}), 503
    
    try:
        data = request.json
        img_str = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image = Image.open(BytesIO(base64.b64decode(img_str))).convert('RGB')
        
        # 1. Detect and Crop Hand
        if hand_detector:
            image = hand_detector.detect_and_crop(image)
        
        # 2. Predict
        raw_letter, raw_conf, probs = model_instance.predict(image)
        updated, stable_letter, stable_conf = stabilizer.update(raw_letter, raw_conf)
        
        top_3 = sorted([{'letter': k, 'confidence': round(v*100, 2)} for k, v in probs.items()], 
                       key=lambda x: x['confidence'], reverse=True)[:3]

        return jsonify({
            'success': True,
            'updated': updated,
            'letter': stable_letter,
            'confidence': round(stable_conf * 100, 2),
            'top_predictions': top_3
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    stabilizer.reset()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)