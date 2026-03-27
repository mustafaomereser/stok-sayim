#!/bin/bash
# ╔══════════════════════════════════════════════════════════╗
# ║   StokSay — EC2 t3.small Kurulum Scripti (Ubuntu 22.04) ║
# ╚══════════════════════════════════════════════════════════╝
# Kullanım: chmod +x setup.sh && sudo ./setup.sh

set -e

echo "── [1/6] Sistem güncelleniyor..."
apt-get update -qq
apt-get install -y python3-pip python3-venv libgl1 libglib2.0-0 nginx -qq

echo "── [2/6] Swap ekleniyor (t3.small için kritik)..."
# 2GB RAM + 2GB swap = bellek baskısını azaltır
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

echo "── [3/6] Uygulama dizini hazırlanıyor..."
mkdir -p /opt/stoksay
cd /opt/stoksay

echo "── [4/6] Python venv ve paketler kuruluyor..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "── [5/6] Systemd servis oluşturuluyor..."
cat > /etc/systemd/system/stoksay.service << 'EOF'
[Unit]
Description=StokSay YOLOv8 API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/stoksay
ExecStart=/opt/stoksay/venv/bin/python main.py
Restart=always
RestartSec=5
# t3.small bellek limiti — aşarsa restart atar
MemoryMax=1500M

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable stoksay

echo "── [6/6] Nginx reverse proxy ayarlanıyor..."
cat > /etc/nginx/sites-available/stoksay << 'EOF'
server {
    listen 80;
    server_name _;

    # Fotoğraf upload boyut limiti
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
echo "Şimdi şunları yap:"
echo "  1. best.pt dosyasını /opt/stoksay/ altına yükle:"
echo "     scp best.pt ubuntu@EC2_IP:/opt/stoksay/"
echo ""
echo "  2. main.py ve requirements.txt'i de yükle:"
echo "     scp main.py ubuntu@EC2_IP:/opt/stoksay/"
echo ""
echo "  3. Servisi başlat:"
echo "     sudo systemctl start stoksay"
echo "     sudo systemctl status stoksay"
echo ""
echo "  4. Test et:"
echo "     curl http://EC2_IP/health"
