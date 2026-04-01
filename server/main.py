"""
StokSay Backend — YOLO + CLIP + DBSCAN
Eğitim yok, referans yok, label yok.
Benzer objeleri otomatik gruplandırır ve sayar.
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
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize as sk_normalize

# PyTorch 2.6 fix
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO

app = FastAPI(title="StokSay API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cpu"

print("YOLO yükleniyor...")
yolo = YOLO("yolov8n.pt")

print("CLIP yükleniyor...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
print("Modeller hazır!")


def get_clip_embedding(img: Image.Image) -> np.ndarray:
    tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = clip_model.encode_image(tensor)
    emb = emb.cpu().numpy().astype(np.float32)
    return sk_normalize(emb)[0]  # (512,)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    confidence: float = Form(default=0.2),   # YOLO eşiği — düşük tut, çok obje yakalasın
    eps: float = Form(default=0.3),           # DBSCAN: ne kadar benzer olunca aynı grup (0-1, küçük = daha katı)
    min_samples: int = Form(default=1),       # DBSCAN: grup için min obje sayısı
):
    """
    1. YOLO → tüm objeleri bul
    2. Her bbox'ı CLIP ile embedding'e çevir
    3. DBSCAN ile benzer embedding'leri grupla
    4. Her grup = ayrı obje türü → say
    """
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

    # ── ADIM 1: YOLO detection ───────────────────────────────────
    yolo_results = yolo.predict(
        frame,
        imgsz=640,
        conf=confidence,
        max_det=300,
        device=DEVICE,
        verbose=False,
    )

    boxes = []
    for r in yolo_results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            bw, bh = x2 - x1, y2 - y1
            if bw < 8 or bh < 8:
                continue
            boxes.append({
                "bbox": [x1, y1, bw, bh],
                "confidence": round(float(box.conf[0]), 3),
            })

    if not boxes:
        return {"detections": [], "elapsed_ms": int((time.time()-t0)*1000), "count": 0}

    # ── ADIM 2: CLIP embedding ───────────────────────────────────
    embeddings = []
    for b in boxes:
        x1, y1, bw, bh = b["bbox"]
        crop = pil_full.crop((x1, y1, x1+bw, y1+bh))
        emb = get_clip_embedding(crop)
        embeddings.append(emb)

    embeddings_np = np.array(embeddings)  # (N, 512)

    # ── ADIM 3: DBSCAN clustering ────────────────────────────────
    # metric=cosine: embedding benzerliği cosine distance ile ölçülür
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    labels = db.fit_predict(embeddings_np)
    # -1 = gürültü (hiçbir gruba uymayan)

    # ── ADIM 4: Sonuçları formatla ───────────────────────────────
    detections = []
    for i, (box, label) in enumerate(zip(boxes, labels)):
        detections.append({
            "group":      int(label),       # -1 = bilinmiyor
            "confidence": box["confidence"],
            "bbox":       box["bbox"],
        })

    elapsed = int((time.time() - t0) * 1000)

    # Grup istatistikleri
    unique_groups = set(l for l in labels if l >= 0)
    group_counts  = {int(g): int(np.sum(labels == g)) for g in unique_groups}

    return {
        "detections":   detections,
        "group_counts": group_counts,   # { 0: 5, 1: 3, 2: 8 }
        "total":        len([l for l in labels if l >= 0]),
        "elapsed_ms":   elapsed,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1, timeout_keep_alive=30)