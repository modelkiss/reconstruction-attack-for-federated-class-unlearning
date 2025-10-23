# ============================================================
#  Dockerfile for Federated Class-Level Unlearning + Attack
#  Environment: Python 3.11, CUDA 11.8, PyTorch 2.1
#  Base: Ubuntu 22.04 (official PyTorch runtime image)
# ============================================================

# === 1. 基础镜像 ===
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# === 2. 基本信息 ===
LABEL maintainer="Weijie Wu <m15906196992@163.com>"
LABEL project="Federated Unlearning & Data Reconstruction Attack"

# === 3. 设置工作目录 ===
WORKDIR /workspace

# === 4. 拷贝代码（可根据实际路径修改） ===
COPY . /workspace

# === 5. 系统依赖 ===
# 包含基本构建工具、OpenCV依赖、git等
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    vim \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# === 6. Python 环境配置 ===
# 确保使用 Python 3.11
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3.11-distutils && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && \
    rm get-pip.py

# === 7. 安装依赖 ===
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip cache purge

# === 8. 环境变量 ===
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/.cache/torch

# === 9. 默认启动命令（进入bash） ===
CMD ["/bin/bash"]

# ============================================================
# Usage:
#   docker build -t fed-unlearning-attack:latest .
#   docker run --gpus all -it -v %cd%/data:/workspace/data -v %cd%/outputs:/workspace/outputs fed-unlearning-attack:latest
# ============================================================
