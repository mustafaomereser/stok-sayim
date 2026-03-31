"""
StokSay Backend — YOLOv8 + CSRNet Hybrid Count Mode
CPU-friendly ve yoğun nesne sayımı için optimize edilmiştir.
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import time
import torch
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms
import torch.nn as nn

# ── CONFIG ────────────────────────────────────────────────────────
YOLO_MODEL_PATH = "yolov8m.pt"
CSRNET_MODEL_PATH = "csrnet_best.pth"  # Eğitimli CSRNet ağırlıkları
IMG_SIZE = 640
PATCH_SIZE = 640
STRIDE = PATCH_SIZE // 2
MAX_DET = 500
DEVICE = "cpu"

# ── FastAPI ───────────────────────────────────────────────────────
app = FastAPI(title="StokSay API - Hybrid Count Mode", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── MODEL YÜKLEME ────────────────────────────────────────────────
print(f"YOLO model yükleniyor: {YOLO_MODEL_PATH}")
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.to(DEVICE)
dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
yolo_model.predict(dummy, imgsz=IMG_SIZE, verbose=False)
print("YOLO hazır!")

# CSRNet basit PyTorch implementasyonu


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        from torchvision import models
        self.frontend_feat = [64, 64, 'M', 128, 128,
                              'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = self._make_layers(self.frontend_feat)
        self.backend = self._make_layers(
            self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if load_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        layers = []
        d_rate = 2 if dilation else 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                                   padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


print(f"CSRNet model yükleniyor: {CSRNET_MODEL_PATH}")
csr_model = CSRNet()
csr_model.load_state_dict(torch.load(CSRNET_MODEL_PATH, map_location=DEVICE))
csr_model.to(DEVICE)
csr_model.eval()
print("CSRNet hazır!")

# ── Yardımcı Fonksiyonlar ─────────────────────────────────────────


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0]+box1[2], box2[0]+box2[2])
    y2 = min(box1[1]+box1[3], box2[1]+box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = box1[2]*box1[3]
    area2 = box2[2]*box2[3]
    union = area1 + area2 - inter
    return inter/union if union > 0 else 0


def filter_overlaps(detections, iou_thresh=0.3):
    filtered = []
    for det in detections:
        if all(iou(det["bbox"], f["bbox"]) < iou_thresh for f in filtered):
            filtered.append(det)
    return filtered


def csr_predict_count(patch):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    img_tensor = transform(patch).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        density_map = csr_model(img_tensor)
        count = density_map.sum().item()
    return count

# ── HEALTH ───────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {
        "status": "ok",
        "yolo_classes": yolo_model.names,
        "device": DEVICE,
    }

# ── DETECT (Hybrid Count Mode) ────────────────────────────────────


@app.post("/detect")
async def detect(image: UploadFile = File(...), confidence: float = Form(default=0.4)):
    t0 = time.time()
    raw = await image.read()
    img_array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return {"detections": [], "elapsed_ms": 0, "count": 0, "error": "Görsel okunamadı"}

    h, w = frame.shape[:2]
    all_detections = []
    csr_total = 0.0

    # ── Patch processing ──────────────────────────────
    for y0 in range(0, h, STRIDE):
        for x0 in range(0, w, STRIDE):
            y1 = min(y0+PATCH_SIZE, h)
            x1 = min(x0+PATCH_SIZE, w)
            patch = frame[y0:y1, x0:x1]

            # YOLO tahmini
            results = yolo_model.predict(
                patch,
                imgsz=IMG_SIZE,
                conf=confidence,
                iou=0.3,
                max_det=MAX_DET,
                device=DEVICE,
                verbose=False
            )
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = yolo_model.names[cls_id]
                    conf_score = float(box.conf[0])
                    x1_box, y1_box, x2_box, y2_box = box.xyxy[0].tolist()
                    global_box = [
                        round(x1_box + x0),
                        round(y1_box + y0),
                        round(x2_box - x1_box),
                        round(y2_box - y1_box)
                    ]
                    all_detections.append({
                        "label": label,
                        "confidence": round(conf_score, 3),
                        "bbox": global_box
                    })

            # CSRNet tahmini
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            csr_count = csr_predict_count(patch_rgb)
            csr_total += csr_count

    # ── Çakışan kutuları filtrele
    filtered_detections = filter_overlaps(all_detections, iou_thresh=0.3)

    # CSRNet ve YOLO toplam sayım
    total_count = len(filtered_detections) + round(csr_total)

    elapsed = int((time.time() - t0) * 1000)

    return {
        "detections": filtered_detections,
        "csr_count": round(csr_total),
        "count": total_count,
        "elapsed_ms": elapsed
    }

# ── RUN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        timeout_keep_alive=30
    )
