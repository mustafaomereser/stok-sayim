from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import clip
from PIL import Image
import numpy as np

app = FastAPI(title="StokSay MultiRef API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# CLIP yükle
DEVICE = "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=DEVICE)

# Referans embeddingleri
reference_embeddings = []
reference_labels = []

@app.post("/references")
async def upload_references(files: list[UploadFile] = File(...)):
    global reference_embeddings, reference_labels
    reference_embeddings = []
    reference_labels = []

    for f in files:
        img = Image.open(f.file).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model_clip.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            reference_embeddings.append(emb)
            reference_labels.append(f.filename)
    return {"status": "ok", "references": reference_labels}

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    img = Image.open(image.file).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_emb = model_clip.encode_image(img_tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        similarities = []
        for ref_emb, label in zip(reference_embeddings, reference_labels):
            sim = (img_emb @ ref_emb.T).item()
            similarities.append({"label": label, "similarity": round(sim,3)})
    return {"similarities": similarities}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)