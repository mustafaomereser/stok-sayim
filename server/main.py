"""
StokSay Backend — YOLOv8 + FastAPI
best.pt varsa onu kullanır, yoksa yolov8n.pt ile çalışır.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import time
import os

# PyTorch 2.6 fix — ultralytics yüklemeden önce yapılmalı
import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────
CUSTOM_MODEL  = "best.pt"       # Eğitilmiş model
FALLBACK_MODEL = "models/yolov8n.pt"   # Yoksa bununla çalış
IMG_SIZE      = 640
MAX_DET       = 300
DEVICE        = "cpu"

# ── MODEL YÜKLEMESİ ──────────────────────────────────────────────
model_path = CUSTOM_MODEL if os.path.exists(CUSTOM_MODEL) else FALLBACK_MODEL
print(f"Model yükleniyor: {model_path}")
model = YOLO(model_path)
model.to(DEVICE)

# Warm-up
dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
model.predict(dummy, imgsz=IMG_SIZE, verbose=False)
print(f"Model hazır! Sınıflar: {list(model.names.values())}")

# ── APP ───────────────────────────────────────────────────────────
app = FastAPI(title="StokSay API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── HEALTH ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":  "ok",
        "model":   model_path,
        "classes": model.names,
        "device":  DEVICE,
    }

# ── DETECT ───────────────────────────────────────────────────────
@app.post("/detect")
async def detect(
    image:      UploadFile = File(...),
    confidence: float      = Form(default=0.4),
):
    """
    Fotoğrafı alır, YOLOv8 ile analiz eder.

    Response:
    {
        "detections": [
            { "label": "somun", "confidence": 0.87, "bbox": [x, y, w, h] }
        ],
        "count":      5,
        "elapsed_ms": 230,
        "image_size": [w, h]
    }
    """
    t0 = time.time()

    raw = await image.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(400, "Görsel okunamadı")

    orig_h, orig_w = frame.shape[:2]

    # Büyük görseli küçült — t3.small için
    if max(orig_h, orig_w) > 1280:
        scale = 1280 / max(orig_h, orig_w)
        frame = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))

    h, w = frame.shape[:2]

    # ── Tahmin ───────────────────────────────────────────────────
    results = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=max(0.1, min(0.95, confidence)),
        max_det=MAX_DET,
        device=DEVICE,
        verbose=False,
    )

    # ── Sonuçları formatla ────────────────────────────────────────
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            bw = x2 - x1
            bh = y2 - y1
            if bw < 5 or bh < 5:
                continue
            detections.append({
                "label":      model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 3),
                "bbox":       [x1, y1, bw, bh],
            })

    elapsed = int((time.time() - t0) * 1000)

    return {
        "detections": detections,
        "count":      len(detections),
        "elapsed_ms": elapsed,
        "image_size": [w, h],
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        timeout_keep_alive=30,
    )