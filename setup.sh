#!/bin/bash
# setup.sh — Ubuntu 24.04 RTX 5090 AI environment bootstrap

set -e

echo "🚀 Updating system..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential python3 python3-venv python3-pip git curl wget

echo "🎮 Installing NVIDIA driver..."
sudo ubuntu-drivers autoinstall

echo "💡 Creating virtual environment..."
python3 -m venv ~/ai-env
source ~/ai-env/bin/activate
pip install --upgrade pip

echo "📦 Installing Python packages..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "Run: source ~/ai-env/bin/activate && python test_gpu.py"

