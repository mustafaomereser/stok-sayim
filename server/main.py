"""
StokSay Backend — YOLOv8 CPU Count Mode (CSRNet removed)
t3.small için optimize edilmiştir.
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import time
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────
MODEL_PATH = "models/yolov8m.pt"  # absolute path
IMG_SIZE = 640
MAX_DET = 500
DEVICE = "cpu"
STRIDE = 320  # patch stride
PATCH_SIZE = 640

# ── FastAPI ───────────────────────────────────────────────────────
app = FastAPI(title="StokSay API - YOLO Count Mode", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── MODEL YÜKLEME ────────────────────────────────────────────────
print(f"YOLO model yükleniyor: {MODEL_PATH}")
yolo_model = YOLO(MODEL_PATH)
yolo_model.to(DEVICE)
dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
yolo_model.predict(dummy, imgsz=IMG_SIZE, verbose=False)
print("YOLO hazır!")

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

# ── HEALTH ───────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {
        "status": "ok",
        "yolo_classes": yolo_model.names,
        "device": DEVICE,
    }

# ── DETECT (YOLO Count Mode) ──────────────────────────────────────


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

    # ── Çakışan kutuları filtrele
    filtered_detections = filter_overlaps(all_detections, iou_thresh=0.3)

    elapsed = int((time.time() - t0) * 1000)
    total_count = len(filtered_detections)

    return {
        "detections": filtered_detections,
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
