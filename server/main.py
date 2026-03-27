"""
StokSay Backend — YOLOv8n + FastAPI
t3.small (2GB RAM, 2 vCPU) için optimize edilmiştir.
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import io
import time
from PIL import Image
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────
MODEL_PATH  = "yolov8n.pt"  # "best.pt"     # Colab'dan indirdiğin dosya
IMG_SIZE    = 640           # Eğitimde kullandığın boyut
MAX_DET     = 100           # Maksimum tespit sayısı
DEVICE      = "cpu"         # t3.small'da GPU yok

app = FastAPI(title="StokSay API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Production'da frontend domain'ini yaz
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── MODEL — tek seferinde yükle ──────────────────────────────────
print(f"Model yükleniyor: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
model.to(DEVICE)
# Warm-up — ilk istek yavaş olmasın
dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
model.predict(dummy, imgsz=IMG_SIZE, verbose=False)
print("Model hazır!")

# ── HEALTH ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "classes": model.names,     # eğitimde kullanılan sınıflar
        "device": DEVICE,
    }

# ── DETECT ───────────────────────────────────────────────────────
@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    confidence: float = Form(default=0.4),
):
    """
    Fotoğrafı alır, YOLOv8n ile analiz eder, tespitleri döner.

    Response:
    {
        "detections": [
            { "label": "somun", "confidence": 0.87, "bbox": [x, y, w, h] }
        ],
        "elapsed_ms": 230,
        "count": 5
    }
    """
    t0 = time.time()

    # Görseli oku
    raw = await image.read()
    img_array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {"detections": [], "elapsed_ms": 0, "count": 0, "error": "Görsel okunamadı"}

    # t3.small için büyük görseli küçült
    h, w = frame.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # Tahmin
    results = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=max(0.1, min(0.95, confidence)),
        max_det=MAX_DET,
        device=DEVICE,
        verbose=False,
    )

    elapsed = int((time.time() - t0) * 1000)

    # Sonuçları formatla
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]
            conf   = float(box.conf[0])
            # xyxy → x, y, w, h (pixel)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "label":      label,
                "confidence": round(conf, 3),
                "bbox":       [round(x1), round(y1), round(x2 - x1), round(y2 - y1)],
            })

    return {
        "detections": detections,
        "elapsed_ms": elapsed,
        "count":      len(detections),
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,          # t3.small'da 1 worker yeterli
        timeout_keep_alive=30,
    )
