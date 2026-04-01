from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import clip
from PIL import Image
import numpy as np

app = FastAPI(title="StokSay MultiRef API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEVICE = "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=DEVICE)

reference_embeddings = []
reference_labels = []

def normalize(tensor):
    return tensor / tensor.norm(dim=-1, keepdim=True)

@app.post("/references")
async def upload_references(files: list[UploadFile] = File(...)):
    global reference_embeddings, reference_labels
    reference_embeddings = []
    reference_labels = []

    for f in files:
        img = Image.open(f.file).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = normalize(model_clip.encode_image(img_tensor))
        reference_embeddings.append(emb)
        reference_labels.append(f.filename)
    return {"status": "ok", "references": reference_labels}

@app.post("/detect")
async def detect(image: UploadFile = File(...), threshold: float = 0.3):
    img = Image.open(image.file).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_emb = normalize(model_clip.encode_image(img_tensor))

    results = []
    for ref_emb, label in zip(reference_embeddings, reference_labels):
        sim = (img_emb @ ref_emb.T).item()
        results.append({"label": label, "similarity": round(sim, 3), "match": sim >= threshold})
    
    # Count kaç tane eşleşiyor
    count = sum(1 for r in results if r["match"])

    return {"results": results, "count": count}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)