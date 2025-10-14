# 🧠 GPU Validation Toolkit — Ubuntu 24.04 + RTX 5090

This repository contains a single Python script, **`test_gpu.py`**,  
for validating your **AI workstation** setup with **NVIDIA RTX 5090**,  
ensuring **CUDA, cuDNN, PyTorch, and TensorFlow** are installed and working properly.

---

## 🚀 Overview

The script checks:

- ✅ System and Python environment info  
- ✅ PyTorch GPU detection and CUDA speed test  
- ✅ TensorFlow GPU detection and compute benchmark  
- ✅ GPU memory allocation and cleanup test  

It’s designed for Ubuntu **24.04 LTS** but works on any modern Linux with CUDA 12+.

---

## ⚙️ 1. System Preparation

Update your system and install build essentials:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential python3 python3-venv python3-pip git curl wget htop
```

2. Install NVidia Driver

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

After reboot

```bash
nvidia-smi
```


3. Install CUDA Toolkit 12.06

https://developer.nvidia.com/cuda-downloads
https://developer.nvidia.com/cuda-12-6-3-download-archive

```bash
Linux → x86_64 → Ubuntu → 24.04 → deb (local)

# copy commands below:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

# add CUDA to your path
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# check version
nvcc --version
# expected: Cuda compilation tools, release 12.6, V12.6.xxx
```

4. Create Python Environment

```bash
python3 -m venv ~/ai-env
source ~/ai-env/bin/activate
pip install --upgrade pip
```

5. Install PyTorch + TensorFlow

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install tensorflow==2.17.0
pip install numpy matplotlib pandas
```

6. Run GPU Validation script

```bash
python test_gpu.py
```

Output:

```
============================================================
PYTORCH TEST
============================================================
PyTorch version: 2.5.0
CUDA available: True
GPU count: 1
GPU 0: NVIDIA GeForce RTX 5090
  Memory: 32.00 GB
GPU matrix multiplication time: 215.67 ms

============================================================
TENSORFLOW TEST
============================================================
TensorFlow version: 2.17.0
GPU devices available: 1
GPU matrix multiplication time: 152.45 ms
Result device: /job:localhost/replica:0/task:0/device:GPU:0

============================================================
MEMORY USAGE
============================================================
GPU memory allocated: 0.00 GB
After allocation - Memory allocated: 0.40 GB
After cleanup - Memory allocated: 0.00 GB
```








