# ╔══════════════════════════════════════════════════════════════╗
# ║         StokSay — YOLOv8n Model Eğitimi (Google Colab)      ║
# ║  Runtime → T4 GPU seç, aksi halde çok yavaş olur            ║
# ╚══════════════════════════════════════════════════════════════╝

# ── ADIM 1: Kurulum ───────────────────────────────────────────────
!pip install ultralytics roboflow -q

# ── ADIM 2: Roboflow'dan dataset indir ───────────────────────────
# roboflow.com → projen → Versions → Export → YOLOv8 formatı
# Aşağıdaki bilgileri kendi projenle doldur

from roboflow import Roboflow

rf = Roboflow(api_key="ROBOFLOW_API_KEY_BURAYA")   # <-- değiştir
project = rf.workspace("WORKSPACE_ADI")             # <-- değiştir
dataset  = project.version(1).download("yolov8")    # <-- version numarası

DATASET_PATH = dataset.location  # otomatik ayarlanır
print("Dataset:", DATASET_PATH)

# ── ADIM 3: Eğitim ───────────────────────────────────────────────
from ultralytics import YOLO

# YOLOv8 nano — t3.small için ideal
model = YOLO("yolov8n.pt")

results = model.train(
    data=f"{DATASET_PATH}/data.yaml",
    epochs=100,          # az fotoğrafta 50-100 yeterli
    imgsz=640,           # görüntü boyutu
    batch=16,            # T4 için 16 iyi, OOM alırsan 8 yap
    patience=20,         # 20 epoch iyileşme olmazsa dur
    optimizer="AdamW",
    lr0=0.001,
    device=0,            # GPU
    project="stoksay",
    name="yolov8n_v1",
    exist_ok=True,
    # Augmentation — az veriyle işe yarar
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

# ── ADIM 4: Sonuçları gör ────────────────────────────────────────
print("\n── Eğitim Tamamlandı ──")
print(f"mAP50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.3f}")

# ── ADIM 5: Modeli test et ───────────────────────────────────────
best_model = YOLO("stoksay/yolov8n_v1/weights/best.pt")

# Validation seti üzerinde değerlendir
val_results = best_model.val(data=f"{DATASET_PATH}/data.yaml")
print("Validation mAP50:", val_results.box.map50)

# ── ADIM 6: ONNX export (opsiyonel ama önerilir — CPU'da daha hızlı) ──
# best_model.export(format="onnx", dynamic=True, simplify=True)
# print("ONNX export tamam: stoksay/yolov8n_v1/weights/best.onnx")

# ── ADIM 7: Modeli indir ─────────────────────────────────────────
from google.colab import files
files.download("stoksay/yolov8n_v1/weights/best.pt")
print("best.pt indirildi — bunu EC2'ye yükle")

# ── NOT: Sınıf isimlerini kontrol et ─────────────────────────────
import yaml
with open(f"{DATASET_PATH}/data.yaml") as f:
    info = yaml.safe_load(f)
print("\nSınıflar:", info["names"])
# Bu listeyi backend'e kopyalaman lazım!
