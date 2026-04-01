#!/bin/bash
# StokSay setup script - t3.small CPU + CLIP

# 1. venv oluştur
python3 -m venv venv
source venv/bin/activate

# 2. pip güncelle
pip install --upgrade pip

# 3. CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. CLIP ve bağımlılıklar
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# 5. FastAPI, Uvicorn, OpenCV, PIL
pip install fastapi uvicorn python-multipart pillow opencv-python

echo "Setup tamam! Venv aktif: 'source venv/bin/activate'"