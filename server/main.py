"""
StokSay Backend — YOLOv8m + FastAPI
t3.small CPU için stable, duplicate-free, tile+IoU merge pipeline
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import time
import torch
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────
MODEL_PATH = "yolov8m.pt"  # medium model → daha doğru tespit
IMG_SIZE = 640
MAX_DET = 200
DEVICE = "cpu"
CONF_MIN = 0.2
IOU_NMS = 0.3       # NMS threshold
TILE_OVERLAP = 100      # overlap px

torch.manual_seed(42)
torch.set_num_threads(1)

app = FastAPI(title="StokSay API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── MODEL LOAD ───────────────────────────────────────────────────
print(f"Model yükleniyor: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
model.to(DEVICE)
dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
model.predict(dummy, imgsz=IMG_SIZE, verbose=False)
print("Model hazır!")

# ── HEALTH ───────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "classes": model.names,
        "device": DEVICE,
    }

# ── TILE FUNCTION ────────────────────────────────────────────────


def tile_image(img, tile_size=IMG_SIZE, overlap=TILE_OVERLAP):
    h, w = img.shape[:2]
    tiles, positions = [], []
    y_steps = list(range(0, h, tile_size - overlap))
    x_steps = list(range(0, w, tile_size - overlap))
    for y in y_steps:
        for x in x_steps:
            y1, x1 = y, x
            y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
            tiles.append(img[y1:y2, x1:x2])
            positions.append((x1, y1))
    return tiles, positions

# ── IOU MERGE ───────────────────────────────────────────────────


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = (x2 - x1)*(y2 - y1) + (x2b - x1b)*(y2b - y1b) - inter
    return inter / (union + 1e-6)


def merge_boxes(boxes, iou_thresh=0.3):
    merged = []
    for box in boxes:
        add = True
        for i, m in enumerate(merged):
            if iou(box, m) > iou_thresh:
                # merge
                mx1 = min(box[0], m[0])
                my1 = min(box[1], m[1])
                mx2 = max(box[2], m[2])
                my2 = max(box[3], m[3])
                merged[i] = [mx1, my1, mx2, my2]
                add = False
                break
        if add:
            merged.append(box)
    return merged

# ── DETECT ───────────────────────────────────────────────────────


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    confidence: float = Form(default=CONF_MIN),
    classes: str = Form(default="person,bottle"),
):
    t0 = time.time()

    raw = await image.read()
    img_array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return {"detections": [], "elapsed_ms": 0, "count": 0, "error": "Görsel okunamadı"}

    tiles, positions = tile_image(frame)
    target_labels = [c.strip() for c in classes.split(",")]
    target_ids = [i for i, n in model.names.items() if n in target_labels]

    all_boxes = []
    all_detections = []

    for tile, (x_offset, y_offset) in zip(tiles, positions):
        results = model.predict(
            tile,
            imgsz=IMG_SIZE,
            conf=max(CONF_MIN, min(0.95, confidence)),
            max_det=MAX_DET,
            iou=IOU_NMS,
            device=DEVICE,
            classes=target_ids,
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
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # global koordinata çevir
                x1 += x_offset
                x2 += x_offset
                y1 += y_offset
                y2 += y_offset
                all_boxes.append([x1, y1, x2, y2])
                all_detections.append({
                    "label": label,
                    "confidence": round(conf_score, 3),
                    "bbox": [round(x1), round(y1), round(x2 - x1), round(y2 - y1)]
                })

    # Duplicate-free merge
    merged_boxes = merge_boxes(all_boxes, iou_thresh=IOU_NMS)
    elapsed = int((time.time() - t0)*1000)

    return {
        "detections": all_detections,
        "elapsed_ms": elapsed,
        "count": len(merged_boxes),
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        timeout_keep_alive=30
    )
