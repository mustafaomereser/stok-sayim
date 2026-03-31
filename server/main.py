"""
StokSay Backend — YOLOv8n + FastAPI (Count Mode)
CPU-friendly ve yoğun nesne sayımı için optimize edilmiştir.
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import io
import time
import torch
from PIL import Image
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────
MODEL_PATH = "yolov8m.pt"  # "best.pt" senin eğitimli modelin
IMG_SIZE = 640              # eğitimde kullanılan boyut
MAX_DET = 500               # CPU’da çok sayıda nesne için arttırıldı
DEVICE = "cpu"

# ── FastAPI ───────────────────────────────────────────────────────
app = FastAPI(title="StokSay API - Count Mode", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── MODEL YÜKLEME ────────────────────────────────────────────────
print(f"Model yükleniyor: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
model.to(DEVICE)
dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
model.predict(dummy, imgsz=IMG_SIZE, verbose=False)
print("Model hazır!")

# ── Yardımcı Fonksiyon: IoU ───────────────────────────────────────


def iou(box1, box2):
    # box = [x, y, w, h]
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

# ── HEALTH ───────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "classes": model.names,
        "device": DEVICE,
    }

# ── DETECT (Count Mode) ───────────────────────────────────────────


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    confidence: float = Form(default=0.4),
):
    t0 = time.time()
    raw = await image.read()
    img_array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {"detections": [], "elapsed_ms": 0, "count": 0, "error": "Görsel okunamadı"}

    # ── Büyük görseli parçalara böl (patch) ────────────────
    h, w = frame.shape[:2]
    PATCH_SIZE = 640
    stride = PATCH_SIZE // 2  # %50 overlap
    all_detections = []

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1 = min(y0 + PATCH_SIZE, h)
            x1 = min(x0 + PATCH_SIZE, w)
            patch = frame[y0:y1, x0:x1]

            results = model.predict(
                patch,
                imgsz=IMG_SIZE,
                conf=confidence,
                iou=0.3,           # yoğun nesneler için düşürdük
                max_det=MAX_DET,
                device=DEVICE,
                verbose=False,
            )

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf_score = float(box.conf[0])
                    x1_box, y1_box, x2_box, y2_box = box.xyxy[0].tolist()
                    # patch koordinatlarını global koordinata çevir
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

    # ── Çakışan kutuları filtrele
    filtered_detections = filter_overlaps(all_detections, iou_thresh=0.3)
    elapsed = int((time.time() - t0) * 1000)

    return {
        "detections": filtered_detections,
        "elapsed_ms": elapsed,
        "count": len(filtered_detections),
    }

# ── RUN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        timeout_keep_alive=30,
    )
