#!/bin/bash
set -e

echo "♻️ TrashMapper V2 Kurulumu Başlıyor..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Sanal ortam oluşturuldu."
fi

source venv/bin/activate

pip install --upgrade pip
pip install streamlit opencv-python-headless onnxruntime numpy matplotlib pyyaml

#sudo apt install python3-libcamera python3-kmsdrm

echo "✅ Kurulum Tamam."
echo "🚀 Başlatılıyor... Tarayıcıdan http://localhost:8501 adresine girin."

streamlit run app.py --server.address 0.0.0.0