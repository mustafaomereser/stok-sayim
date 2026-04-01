"""
StokSay Backend — YOLO + CLIP + Tek Referans
Kullanıcı fotoğraftan bir obje seçer,
sistem benzerlerini bulup sayar. iScanner mantığı.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import clip
from PIL import Image
import numpy as np
import cv2
import time
import io

# PyTorch 2.6 fix
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO

app = FastAPI(title="StokSay API", version="5.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cpu"

print("YOLO yükleniyor...")
yolo = YOLO("models/yolov8n.pt")

print("CLIP yükleniyor...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
print("Modeller hazır!")


def normalize(t):
    return t / t.norm(dim=-1, keepdim=True)

def get_embedding(img: Image.Image) -> torch.Tensor:
    tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = clip_model.encode_image(tensor)
    return normalize(emb)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(
    # Ana fotoğraf
    image: UploadFile = File(...),
    # Kullanıcının seçtiği referans crop (tek obje)
    reference: UploadFile = File(...),
    # YOLO güven eşiği
    confidence: float = Form(default=0.2),
    # CLIP benzerlik eşiği — düşük = daha fazla eşleşir
    similarity: float = Form(default=0.55),
):
    t0 = time.time()

    # ── Ana görseli oku ──────────────────────────────────────────
    raw = await image.read()
    img_array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Ana görsel okunamadı")

    orig_h, orig_w = frame.shape[:2]

    # Büyük görseli küçült
    if max(orig_h, orig_w) > 1280:
        scale = 1280 / max(orig_h, orig_w)
        frame = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))

    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h, w = frame.shape[:2]

    # ── Referans crop embedding ──────────────────────────────────
    ref_raw = await reference.read()
    ref_img = Image.open(io.BytesIO(ref_raw)).convert("RGB")
    ref_emb = get_embedding(ref_img)  # (1, 512)

    # ── YOLO ile tüm objeleri bul ────────────────────────────────
    yolo_results = yolo.predict(
        frame,
        imgsz=640,
        conf=confidence,
        max_det=300,
        device=DEVICE,
        verbose=False,
    )

    detections = []
    for r in yolo_results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            bw, bh = x2 - x1, y2 - y1
            if bw < 8 or bh < 8:
                continue

            # Crop et
            crop = pil_frame.crop((x1, y1, x2, y2))
            crop_emb = get_embedding(crop)

            # Referansa benzerlik
            sim = (crop_emb @ ref_emb.T).item()

            if sim >= similarity:
                detections.append({
                    "bbox":       [x1, y1, bw, bh],
                    "similarity": round(sim, 3),
                    "confidence": round(float(box.conf[0]), 3),
                })

    # Benzerliğe göre sırala
    detections.sort(key=lambda x: x["similarity"], reverse=True)

    elapsed = int((time.time() - t0) * 1000)

    return {
        "detections": detections,
        "count":      len(detections),
        "elapsed_ms": elapsed,
        "image_size": [w, h],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1, timeout_keep_alive=30)