#!/bin/bash
# ============================================
# SETUP MÔI TRƯỜNG — Mỗi thành viên chạy 1 lần
# Cách dùng: bash setup_env.sh
# ============================================

set -e

echo "============================================"
echo "  TCB Prediction — Setup môi trường"
echo "============================================"

# Kiểm tra Python
python3 --version || { echo "❌ Cần cài Python 3.10+"; exit 1; }

# Tạo virtual environment
echo ""
echo ">>> Tạo virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Cài thư viện
echo ""
echo ">>> Cài đặt thư viện..."
pip install --upgrade pip
pip install -r requirements.txt

# Tạo thư mục cần thiết
mkdir -p models/saved logs notebooks

# Tạo database
echo ""
echo ">>> Tạo database..."
python -m database.schema

echo ""
echo "============================================"
echo "  ✅ Setup hoàn tất!"
echo "============================================"
echo ""
echo "  Kích hoạt môi trường mỗi khi làm việc:"
echo "    source venv/bin/activate"
echo ""
echo "  Kiểm tra database:"
echo "    python -m database.schema"
echo ""
