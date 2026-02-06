#!/bin/bash
set -e

echo "♻️ TrashMapper V2 Installation Begins..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "A virtual environment has been created."
fi

source venv/bin/activate

pip install --upgrade pip
pip install streamlit opencv-python-headless onnxruntime numpy matplotlib pyyaml

#sudo apt install python3-libcamera python3-kmsdrm

echo "✅ Installation OK."
echo "🚀 Initializing... From browser http://localhost:8501 "

streamlit run app.py --server.address 0.0.0.0
