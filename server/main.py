"""
StokSay Backend — YOLO (detection) + CLIP (classification)
Eğitim yok, referans fotoğraflardan öğrenir.
t3.small (2GB RAM, 2 vCPU) için optimize edilmiştir.
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

app = FastAPI(title="StokSay API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cpu"

# ── MODEL YÜKLEMESİ ──────────────────────────────────────────────
print("YOLO yükleniyor...")
yolo = YOLO("yolov8n.pt")

print("CLIP yükleniyor...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
print("Modeller hazır!")

# ── REFERANS STORE ────────────────────────────────────────────────
# { "somun": [emb1, emb2, ...], "röle": [...] }
reference_store: dict[str, list[torch.Tensor]] = {}

def normalize(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(dim=-1, keepdim=True)

def get_clip_embedding(img: Image.Image) -> torch.Tensor:
    tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = clip_model.encode_image(tensor)
    return normalize(emb)

def get_mean_embedding(label: str) -> torch.Tensor:
    embs = reference_store[label]
    stacked = torch.cat(embs, dim=0)
    return normalize(stacked.mean(dim=0, keepdim=True))

# ── HEALTH ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "references": {k: len(v) for k, v in reference_store.items()},
        "device": DEVICE,
    }

# ── REFERANS YÜKLEMESİ ───────────────────────────────────────────
@app.post("/references")
async def upload_references(
    files: list[UploadFile] = File(...),
    label: str = Form(...),
):
    """
    Bir ürün için referans fotoğrafları yükle.
    label: ürün adı (ör: "somun")
    files: o ürünün 3-10 fotoğrafı
    Birden fazla ürün için birden fazla kez çağır.
    """
    embeddings = []
    for f in files:
        raw = await f.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        emb = get_clip_embedding(img)
        embeddings.append(emb)

    reference_store[label] = embeddings
    return {
        "status": "ok",
        "label": label,
        "photo_count": len(embeddings),
        "all_labels": list(reference_store.keys()),
    }

@app.get("/references")
def list_references():
    return {"references": {k: len(v) for k, v in reference_store.items()}}

@app.delete("/references/{label}")
def delete_reference(label: str):
    if label not in reference_store:
        raise HTTPException(404, "Referans bulunamadı")
    del reference_store[label]
    return {"status": "ok", "remaining": list(reference_store.keys())}

# ── DETECT ───────────────────────────────────────────────────────
@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    confidence: float = Form(default=0.3),  # YOLO eşiği
    similarity: float = Form(default=0.6),  # CLIP benzerlik eşiği
):
    """
    1. YOLO → tüm objeleri bul (bbox)
    2. Her bbox'ı crop et
    3. CLIP → referans fotoğraflarına benzerliğe göre sınıflandır
    4. Eşleşenleri say ve döndür

    Response:
    {
        "detections": [
            { "label": "somun", "confidence": 0.87, "similarity": 0.72, "bbox": [x, y, w, h] }
        ],
        "elapsed_ms": 430,
        "count": 5
    }
    """
    if not reference_store:
        raise HTTPException(400, "Önce /references ile referans fotoğraf yükle")

    t0 = time.time()

    raw = await image.read()
    img_array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Görsel okunamadı")

    # Büyük görseli küçült
    h, w = frame.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    pil_full = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Referans embeddingler
    label_embeddings = {l: get_mean_embedding(l) for l in reference_store}

    # YOLO detection
    yolo_results = yolo.predict(frame, imgsz=640, conf=confidence, max_det=200, device=DEVICE, verbose=False)

    detections = []
    for r in yolo_results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            bw, bh = x2 - x1, y2 - y1
            if bw < 10 or bh < 10:
                continue

            crop = pil_full.crop((x1, y1, x2, y2))
            crop_emb = get_clip_embedding(crop)

            # En benzer label'ı bul
            best_label, best_sim = None, -1.0
            for label, ref_emb in label_embeddings.items():
                sim = (crop_emb @ ref_emb.T).item()
                if sim > best_sim:
                    best_sim, best_label = sim, label

            if best_sim < similarity:
                continue

            detections.append({
                "label":      best_label,
                "confidence": round(float(box.conf[0]), 3),
                "similarity": round(best_sim, 3),
                "bbox":       [x1, y1, bw, bh],
            })

    elapsed = int((time.time() - t0) * 1000)
    return {"detections": detections, "elapsed_ms": elapsed, "count": len(detections)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1, timeout_keep_alive=30)