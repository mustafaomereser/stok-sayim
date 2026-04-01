from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import time
from PIL import Image
import torch
import clip  # pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git
import os

# ── CONFIG ───────────────────────────────────────────────────────
DEVICE = "cpu"
PATCH_SIZE = 640
STRIDE = 320
SIMILARITY_THRESH = 0.75
MAX_REFERENCES = 5

# ── FastAPI ───────────────────────────────────────────────────────
app = FastAPI(title="StokSay Multi-Reference Count", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── CLIP MODEL ────────────────────────────────────────────────────
print("CLIP model yükleniyor...")
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
print("CLIP hazır!")

# ── Referans embeddingleri ─────────────────────────────────────────
reference_embeddings = []
reference_labels = []


@app.post("/references")
async def upload_references(files: list[UploadFile] = File(...)):
    global reference_embeddings, reference_labels
    reference_embeddings = []
    reference_labels = []

    if len(files) > MAX_REFERENCES:
        return {"error": f"En fazla {MAX_REFERENCES} referans yükleyebilirsin."}

    for f in files:
        raw = await f.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_input = preprocess(img).unsqueeze(0)  # batch 1
        with torch.no_grad():
            emb = clip_model.encode_image(img_input.to(DEVICE))
            emb /= emb.norm(dim=-1, keepdim=True)  # normalize
        reference_embeddings.append(emb)
        reference_labels.append(f.filename)

    return {"status": "ok", "uploaded": [f.filename for f in files]}

# ── YARDIMCI FONKSİYON ─────────────────────────────────────────────


def cosine_similarity(a, b):
    return (a @ b.T).item()


def filter_overlaps(detections, iou_thresh=0.3):
    filtered = []
    for det in detections:
        if all(iou(det["bbox"], f["bbox"]) < iou_thresh for f in filtered):
            filtered.append(det)
    return filtered


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

# ── DETECT ───────────────────────────────────────────────────────


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    if not reference_embeddings:
        return {"error": "Önce referans fotoğrafları yüklemelisin."}

    t0 = time.time()
    raw = await image.read()
    img_array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return {"detections": [], "count": 0, "elapsed_ms": 0, "error": "Görsel okunamadı"}

    h, w = frame.shape[:2]
    all_detections = []

    # ── PATCH TARAMA ───────────────────────────────────────────────
    for y0 in range(0, h, STRIDE):
        for x0 in range(0, w, STRIDE):
            y1 = min(y0+PATCH_SIZE, h)
            x1 = min(x0+PATCH_SIZE, w)
            patch = frame[y0:y1, x0:x1]
            img_patch = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            img_input = preprocess(img_patch).unsqueeze(0)

            with torch.no_grad():
                patch_emb = clip_model.encode_image(img_input.to(DEVICE))
                patch_emb /= patch_emb.norm(dim=-1, keepdim=True)

            # ── REFERANS KARŞILAŞTIRMA ─────────────────────────
            for ref_emb, label in zip(reference_embeddings, reference_labels):
                sim = cosine_similarity(patch_emb, ref_emb)
                if sim >= SIMILARITY_THRESH:
                    all_detections.append({
                        "label": label,
                        "confidence": round(sim, 3),
                        "bbox": [x0, y0, x1-x0, y1-y0]
                    })

    filtered_detections = filter_overlaps(all_detections, iou_thresh=0.3)
    elapsed = int((time.time() - t0) * 1000)

    return {
        "detections": filtered_detections,
        "count": len(filtered_detections),
        "elapsed_ms": elapsed
    }

# ── RUN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)
