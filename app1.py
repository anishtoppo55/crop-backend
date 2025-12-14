from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# ---------------------------------------------------------
# 1. DEFINE MODEL ARCHITECTURE (from your training notebook)
# ---------------------------------------------------------

class GatedFusionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, image_features, sift_features):
        combined = torch.cat([image_features, sift_features], dim=1)
        attention_map = self.gate_conv(combined)
        return image_features + (sift_features * attention_map)


class TwoStreamSIFTNet(nn.Module):
    def __init__(self, num_classes, sift_channels=128):
        super().__init__()

        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_stream = nn.Sequential(*list(resnet.children())[:7])

        self.sift_stream = nn.Sequential(
            nn.Conv2d(sift_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fusion_gate = GatedFusionModule(in_channels=256)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, rgb, sift):
        f1 = self.image_stream(rgb)
        f2 = self.sift_stream(sift)
        fused = self.fusion_gate(f1, f2)
        return self.classifier(fused)

# -----------------------------------
# 2. LOAD MODEL + CLASSES
# -----------------------------------

CLASSES = ["black gram", "chickpea", "corn", "groundnut",
           "millets", "mustard", "pegeon pea", "soyabean", "wheat"]
NUM_CLASSES = len(CLASSES)

model = TwoStreamSIFTNet(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# -----------------------------------
# 3. PREPROCESSING (RGB + DENSE SIFT)
# -----------------------------------

IMG_SIZE = (224, 224)
SIFT_STRIDE = 8
SIFT_KEYPOINT_SIZE = 16

def extract_dense_sift(image_bgr):
    sift = cv2.SIFT_create()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    keypoints = [
        cv2.KeyPoint(x, y, SIFT_KEYPOINT_SIZE)
        for y in range(0, gray.shape[0], SIFT_STRIDE)
        for x in range(0, gray.shape[1], SIFT_STRIDE)
    ]

    _, descriptors = sift.compute(gray, keypoints)
    if descriptors is None:
        return None

    h = gray.shape[0] // SIFT_STRIDE
    w = gray.shape[1] // SIFT_STRIDE

    sift_map = descriptors.reshape(h, w, 128)
    sift_map = cv2.resize(sift_map, IMG_SIZE)
    return sift_map


def preprocess_for_model(img_pil):
    # Convert PIL → BGR numpy (OpenCV)
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    img_bgr = cv2.resize(img_bgr, IMG_SIZE)

    # Extract SIFT features
    sift_map = extract_dense_sift(img_bgr)
    if sift_map is None:
        raise ValueError("SIFT feature extraction failed.")

    # Convert RGB & SIFT to tensors
    rgb = torch.from_numpy(img_bgr).permute(2, 0, 1).float() / 255.0
    sift = torch.from_numpy(sift_map).permute(2, 0, 1).float()

    return rgb.unsqueeze(0), sift.unsqueeze(0)

# -----------------------------------
# 4. FLASK API
# -----------------------------------

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Crop classification backend is running 🚀"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "image_url missing"}), 400

    # Download image
    resp = requests.get(image_url)
    img = Image.open(BytesIO(resp.content)).convert("RGB")

    try:
        rgb, sift = preprocess_for_model(img)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    with torch.no_grad():
        output = model(rgb, sift)
        pred = output.argmax().item()

    return jsonify({
        "crop_type": CLASSES[pred],
        "class_index": pred
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

