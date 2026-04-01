# ╔══════════════════════════════════════════════════════════════════╗
# ║        StokSay — YOLOv8n Eğitim Notebook (Google Colab)         ║
# ║  Çalıştırmadan önce: Runtime → T4 GPU seç                       ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── ADIM 1: Kurulum ───────────────────────────────────────────────
#!pip install ultralytics roboflow -q

# ── ADIM 2: Roboflow'dan dataset indir ───────────────────────────
# roboflow.com → projen → Versions → Export → YOLOv8 formatı
from roboflow import Roboflow

RF_API_KEY   = "BURAYA_API_KEY"    # Roboflow API key
RF_WORKSPACE = "BURAYA_WORKSPACE"  # workspace adı
RF_PROJECT   = "BURAYA_PROJE"      # proje adı
RF_VERSION   = 1                   # dataset versiyonu

rf = Roboflow(api_key=RF_API_KEY)
project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
dataset = project.version(RF_VERSION).download("yolov8")

DATASET_PATH = dataset.location
print("Dataset:", DATASET_PATH)

# ── ADIM 3: Eğitim ───────────────────────────────────────────────
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # nano — t3.small için ideal

results = model.train(
    data=f"{DATASET_PATH}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,          # OOM alırsan 8 yap
    patience=20,       # 20 epoch iyileşme olmazsa dur
    optimizer="AdamW",
    lr0=0.001,
    device=0,          # GPU
    project="stoksay",
    name="v1",
    exist_ok=True,
    # Augmentation
    flipud=0.3,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
)

# ── ADIM 4: Sonuçlar ─────────────────────────────────────────────
print("\n── Eğitim Tamamlandı ──")
print(f"mAP50:    {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.3f}")

# ── ADIM 5: Validation ───────────────────────────────────────────
best = YOLO("stoksay/v1/weights/best.pt")
val  = best.val(data=f"{DATASET_PATH}/data.yaml")
print("Val mAP50:", val.box.map50)

# Sınıfları yazdır — bunları server/main.py'e kopyala
import yaml
with open(f"{DATASET_PATH}/data.yaml") as f:
    info = yaml.safe_load(f)
print("\nSınıflar:", info["names"])

# ── ADIM 6: İndir ────────────────────────────────────────────────
from google.colab import files
files.download("stoksay/v1/weights/best.pt")
print("best.pt indirildi!")

# EC2'ye yükle:
# scp best.pt ubuntu@EC2_IP:/home/ubuntu/stok-sayim/server/
# sudo systemctl restart stoksay