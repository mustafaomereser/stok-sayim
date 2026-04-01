# ╔══════════════════════════════════════════════════════════════════╗
# ║        StokSay — YOLOv8n Eğitim (Google Colab)                  ║
# ║  Runtime → T4 GPU seç önce!                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── HÜCRE 1: Kurulum ──────────────────────────────────────────────
!pip install ultralytics -q

# ── HÜCRE 2: Zip yükle ve aç ─────────────────────────────────────
from google.colab import files
import zipfile, os, shutil

print("Zip dosyasını seç...")
uploaded = files.upload()
zip_name = list(uploaded.keys())[0]

os.makedirs("/content/dataset", exist_ok=True)
with zipfile.ZipFile(zip_name, 'r') as z:
    z.extractall("/content/dataset")

print("Zip açıldı:")
for root, dirs, fs in os.walk("/content/dataset"):
    for f in fs[:5]:
        print(" ", os.path.join(root, f))

# ── HÜCRE 3: Train/Valid split ────────────────────────────────────
import random

BASE      = "/content/dataset"
TRAIN     = f"{BASE}/images/train"
VALID     = f"{BASE}/images/valid"
LBL_TRAIN = f"{BASE}/labels/train"
LBL_VALID = f"{BASE}/labels/valid"

for d in [TRAIN, VALID, LBL_TRAIN, LBL_VALID]:
    os.makedirs(d, exist_ok=True)

img_dir = f"{BASE}/images"
lbl_dir = f"{BASE}/labels"

images = [
    f for f in os.listdir(img_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'))
    and os.path.isfile(f"{img_dir}/{f}")
]
random.seed(42)
random.shuffle(images)

split      = int(len(images) * 0.8)
train_imgs = images[:split]
valid_imgs = images[split:]

def copy_pair(imgs, img_dst, lbl_dst):
    for img in imgs:
        lbl = os.path.splitext(img)[0] + ".txt"
        src_img = f"{img_dir}/{img}"
        src_lbl = f"{lbl_dir}/{lbl}"
        if os.path.exists(src_img):
            shutil.copy(src_img, f"{img_dst}/{img}")
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, f"{lbl_dst}/{lbl}")

copy_pair(train_imgs, TRAIN, LBL_TRAIN)
copy_pair(valid_imgs, VALID, LBL_VALID)

print(f"Train: {len(train_imgs)} görsel")
print(f"Valid: {len(valid_imgs)} görsel")

# ── HÜCRE 4: data.yaml ───────────────────────────────────────────
with open(f"{BASE}/classes.txt") as f:
    classes = [line.strip() for line in f if line.strip()]

yaml_content = f"""path: {BASE}
train: images/train
val:   images/valid

nc: {len(classes)}
names: {classes}
"""

with open(f"{BASE}/data.yaml", "w") as f:
    f.write(yaml_content)

print("data.yaml:")
print(yaml_content)

# ── HÜCRE 5: Eğitim ──────────────────────────────────────────────
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

results = model.train(
    data=f"{BASE}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    patience=20,
    optimizer="AdamW",
    lr0=0.001,
    device=0,
    project="/content/stoksay",
    name="v1",
    exist_ok=True,
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

print(f"\nmAP50:    {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.3f}")

# ── HÜCRE 6: İndir ───────────────────────────────────────────────
from google.colab import files
files.download("/content/stoksay/v1/weights/best.pt")
print("best.pt indirildi!")
print()
print("EC2'ye yükle:")
print("  scp best.pt ubuntu@EC2_IP:/home/ubuntu/stok-sayim/server/")
print("  sudo systemctl restart stoksay")