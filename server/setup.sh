#!/bin/bash
# Kullanım: sudo bash setup.sh

set -e

REPO="https://github.com/mustafaomereser/stok-sayim"
APP_DIR="/home/ubuntu/stok-sayim"

echo "── [1/4] Sistem güncelleniyor..."
apt-get update -qq
apt-get install -y python3-pip python3-venv libgl1 libglib2.0-0 nginx git -qq

echo "── [2/4] Swap kontrol ediliyor..."
if swapon --show | grep -q '/'; then
  echo "   Swap zaten aktif, atlanıyor."
else
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  echo "   2GB swap aktif."
fi

echo "── [3/4] Repo clone / güncelleniyor..."
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull
else
  git clone "$REPO" "$APP_DIR"
fi

echo "── [4/4] Python venv ve paketler kuruluyor..."
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip -q
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/server/requirements.txt" -q

echo "── Systemd servisi ayarlanıyor..."
cat > /etc/systemd/system/stoksay.service << EOF
[Unit]
Description=StokSay YOLOv8 API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$APP_DIR/server
ExecStart=$APP_DIR/venv/bin/python main.py
Restart=always
RestartSec=5
MemoryMax=1500M

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable stoksay

echo "── Nginx ayarlanıyor..."
cat > /etc/nginx/sites-available/stoksay << 'EOF'
server {
    listen 80;
    server_name _;
    client_max_body_size 10M;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 60s;
    }
}
EOF

ln -sf /etc/nginx/sites-available/stoksay /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

echo ""
echo "✅ Kurulum tamam!"
echo ""
echo "best.pt modelini yükle sonra:"
echo "  scp best.pt ubuntu@EC2_IP:$APP_DIR/server/"
echo ""
echo "Servisi başlat:"
echo "  sudo systemctl start stoksay"
echo "  curl http://EC2_IP/health"