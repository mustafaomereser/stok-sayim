#!/bin/bash
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

[ "$EUID" -ne 0 ] && error "sudo ile çalıştırın: sudo bash setup.sh"

# ── Konfig ───────────────────────────────────────────────────────
read -p "Domain (örn: stoksay.aracservistakip.com): " DOMAIN
read -p "SSL e-posta: " EMAIL
# ─────────────────────────────────────────────────────────────────

REPO="https://github.com/mustafaomereser/stok-sayim"
APP_DIR="/home/ubuntu/stok-sayim"
WEB_DIR="/var/www/stoksay"

info "Sistem güncelleniyor..."
apt-get update -qq
apt-get install -y python3-pip python3-venv libgl1 libglib2.0-0 nginx git certbot python3-certbot-nginx -qq
success "Sistem güncellendi"

info "Swap kontrol ediliyor..."
if swapon --show | grep -q '/'; then
  warn "Swap zaten aktif, atlanıyor."
else
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  success "2GB swap aktif."
fi

info "Repo clone / güncelleniyor..."
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull
  success "Repo güncellendi"
else
  git clone "$REPO" "$APP_DIR"
  success "Repo clone'landı"
fi

info "Python venv ve paketler kuruluyor..."
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip -q
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/server/requirements.txt" -q
success "Python paketleri kuruldu"

info "Frontend dosyaları kopyalanıyor..."
mkdir -p "$WEB_DIR"
cp "$APP_DIR/index.html" "$WEB_DIR/index.html"
success "Frontend hazır"

info "Systemd servisi kuruluyor..."
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
systemctl restart stoksay
success "Systemd servisi aktif"

info "Nginx HTTP yapılandırılıyor..."
cat > /etc/nginx/sites-available/stoksay << EOF
server {
    listen 80;
    server_name $DOMAIN;
    client_max_body_size 10M;

    # Frontend
    root $WEB_DIR;
    index index.html;
    location / {
        try_files \$uri \$uri/ =404;
    }

    # API — tüm endpoint'leri proxy'le
    location ~ ^/(detect|health|references) {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host \$host;
        proxy_set_header   X-Real-IP \$remote_addr;
        proxy_read_timeout 60s;
    }
}
EOF

ln -sf /etc/nginx/sites-available/stoksay /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx
success "Nginx aktif"

info "SSL sertifikası alınıyor..."
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" --redirect
success "SSL kuruldu"

systemctl restart nginx

info "Health check..."
sleep 5
HEALTH=$(curl -s http://127.0.0.1:8000/health)
if echo "$HEALTH" | grep -q "ok"; then
  success "API çalışıyor"
else
  warn "API henüz hazır değil, kontrol et: sudo journalctl -u stoksay -n 30"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Kurulum tamamlandı!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Frontend : https://$DOMAIN"
echo "  API      : https://$DOMAIN/detect"
echo "  Health   : https://$DOMAIN/health"
echo "  Loglar   : sudo journalctl -u stoksay -f"
echo ""