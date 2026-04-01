"""
StokSay Backend — Saf OpenCV ile obje sayma
AI yok, model yok, eğitim yok.
Kontur tespiti ile objeleri bulur ve sayar.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import time

app = FastAPI(title="StokSay API", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "mode": "opencv-contour"}


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    # Minimum obje alanı (piksel²) — küçük gürültüleri eler
    min_area: int = Form(default=200),
    # Maksimum obje alanı — çok büyük alanları eler (arka plan vs)
    max_area: int = Form(default=50000),
    # Blur miktarı — gürültüyü azaltır (tek sayı, çift olursa +1)
    blur: int = Form(default=5),
    # Threshold tipi: "otsu", "adaptive"
    threshold_type: str = Form(default="otsu"),
):
    t0 = time.time()

    raw = await image.read()
    img_array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Görsel okunamadı")

    h, w = frame.shape[:2]

    # ── Ön işleme ────────────────────────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur — çift olursa +1 yap (OpenCV gereksinimi)
    b = blur if blur % 2 == 1 else blur + 1
    blurred = cv2.GaussianBlur(gray, (b, b), 0)

    # Threshold
    if threshold_type == "adaptive":
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
    else:  # otsu
        _, thresh = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    # Morfoloji — küçük delikleri kapat, gürültüyü temizle
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

    # ── Kontur tespiti ────────────────────────────────────────────
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Çok uzun/ince şekilleri ele — muhtemelen çizgi/gürültü
        aspect = bw / bh if bh > 0 else 0
        if aspect > 8 or aspect < 0.125:
            continue

        # Konturun merkezi
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + bw // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + bh // 2

        detections.append({
            "bbox":   [x, y, bw, bh],
            "center": [cx, cy],
            "area":   int(area),
        })

    elapsed = int((time.time() - t0) * 1000)

    return {
        "detections": detections,
        "count":      len(detections),
        "elapsed_ms": elapsed,
        "image_size": [w, h],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1, timeout_keep_alive=30)