Federated Class-Level Unlearning + Data Reconstruction Attack
🚀 项目简介

本项目用于研究联邦学习中类级遗忘后的数据重建攻击，包含：

类级遗忘算法实现（基于 FedAvg / FedOpt 等）

标签推理（置信度差异 + 热力图差异）

数据重建（基于公共数据扰动）

可视化与评估工具（Heatmap / Confidence Shift / Reconstruction Metrics）

模型：ResNet、LeNet
数据集：CIFAR-10、CIFAR-100、MNIST、FashionMNIST

🐳 使用 Docker 快速运行环境
💡 为什么使用 Docker？

Docker 提供：

✅ 统一环境（不依赖本地 Python / CUDA 版本）

✅ GPU 支持（通过 --gpus all 参数）

✅ 快速复现（任何人都能一键运行你的实验）

✅ 干净隔离（不污染本地 Python 环境）

🧱 1. 构建镜像

确保已安装：

Docker Desktop for Windows

NVIDIA Container Toolkit
（用于 GPU 支持）
↳

在项目根目录（含有 Dockerfile 的位置）打开 PowerShell 或 CMD：

docker build -t fed-unlearning-attack:latest .


⚙️ 这将基于 Ubuntu + CUDA 11.8 + Python 3.11 构建完整实验环境。

🧩 2. 启动容器

使用以下命令运行容器（并挂载本地数据与输出文件夹）：

docker run --gpus all -it ^
  -v %cd%/data:/workspace/data ^
  -v %cd%/outputs:/workspace/outputs ^
  --name fedunlearn ^
  fed-unlearning-attack:latest


说明：

--gpus all：启用 GPU（需支持 CUDA）

-v：挂载本地文件夹（Windows 下 %cd% 代表当前路径）

--name fedunlearn：容器命名，方便后续管理

进入容器后，你将在 /workspace 下看到所有源代码文件。

🧠 3. 运行实验脚本

示例命令：

# 训练初始联邦模型
python src/train_federated.py --config configs/cifar10_resnet.yaml

# 执行类级遗忘
python src/unlearning_class.py --config configs/unlearn_cifar10.yaml

# 执行数据重建攻击
python src/reconstruction_attack.py --config configs/attack_cifar10.yaml

# 生成热力图与置信度分析
python src/visualization/heatmap_analysis.py --input outputs/models/

💾 4. 输出目录结构

默认输出在 /workspace/outputs（已挂载到本地），包括：

outputs/
├── models/               # 模型权重（before / after unlearning）
├── heatmaps/             # 热力图（per class / per sample）
├── logs/                 # 训练与攻击日志
└── reconstructions/      # 数据重建结果（图像）

🔧 5. 常用 Docker 命令
# 查看正在运行的容器
docker ps

# 停止容器
docker stop fedunlearn

# 重新进入容器
docker exec -it fedunlearn /bin/bash

# 删除容器
docker rm fedunlearn

# 删除镜像
docker rmi fed-unlearning-attack:latest

🧩 文件结构建议
project_root/
├── README.md
├── requirements.txt
├── environment.yml         # 可选: conda environment file
├── docker/                 # 可选: Dockerfile 与说明
│   └── Dockerfile
├── configs/                # YAML/JSON 配置文件（实验参数）
│   ├── default.yaml
│   ├── datasets
│   │   ├── cifar10.yaml
│   │   ├── cifar100.yaml
│   │   ├── fashionmnist.yaml
│   │   └── mnist.yaml
│   ├── experiments
│   │   ├── cifar10_resnet20_fedavg.yaml
│   │   ├── cifar100_resnet20_fedopt.yaml
│   │   ├── fashionmnist_lenet_fedavg.yaml
│   │   └── mnist_lenet_fedavg.yaml
│   ├── models
│   │   ├── resnet20.yaml
│   │   └── lenet.yaml
│   ├── strategies
│   │   ├── fedavg.yaml
│   │   ├── fedopt.yaml
│   │   └── fedprox.yaml
│   ├── attack_default.yaml
│   └── unlearning_default.yaml
├── data/                   # 原始下载的数据（只放下载脚本或少量样本）
│   ├── raw/
│   └── processed/          # 预处理/缓存后的数据（按数据集分）
│       ├── CIFAR10/
│       │   ├── train.pt
│       │   ├── val.pt
│       │   └── meta.json
│       └── MNIST/...
├── scripts/                # 辅助脚本（下载数据、评估脚本、例行任务）
│   ├── download_datasets.py
│   ├── preprocess_dataset.py
│   └── evaluate_oracle.py
├── src/                    # 源码主目录
│   ├── __init__.py
│   ├── utils/              # 各类工具函数、IO、logging、seeds、metrics
│   │   ├── io.py
│   │   ├── logging.py
│   │   ├── seeds.py
│   │   ├── metrics.py
│   │   └── viz.py          # 画图、heatmap 存储等
│   ├── data/               # Dataset / Dataloader / transforms / caching
│   │   ├── dataset_factory.py
│   │   └── transforms.py
│   ├── models/             # model 构建（resnet, lenet, small CNN）
│   │   ├── resnet.py
│   │   ├── lenet.py
│   │   └── model_utils.py
│   ├── federated/          # 联邦训练与服务器/客户端模拟
│   │   ├── server.py
│   │   ├── client.py
│   │   └── strategies.py   # FedAvg, FedOpt, FedProx 的实现/接口
│   ├── unlearning/         # 遗忘模块：不同遗忘策略的实现（接口化）
│   │   ├── base_unlearner.py
│   │   ├── retrain_unlearner.py
│   │   └── approximate_unlearner.py
│   ├── attack/             # 攻击模块：标签推理 + 数据重建（高层接口）
│   │   ├── label_inference.py
│   │   ├── reconstruction.py
│   │   └── attack_utils.py
│   ├── explainability/     # 生成热力图/Saliency map（captum wrappers）
│   │   └── saliency.py
│   └── experiments/        # experiment runner(s): orchestrate runs, logging, checkpoints
│       ├── run_federated.py
│       ├── run_unlearning.py
│       └── run_attack.py
├── outputs/                # 所有实验输出（按实验/时间戳分）
│   ├── experiments/        # 实验级文件夹（每次 run 一个文件夹）
│   │   └── 2025-10-20__exp001__cifar10_resnet20_fedavg/
│   │       ├── config.yaml
│   │       ├── checkpoints/
│   │       │   ├── model_before.pth
│   │       │   └── model_after.pth
│   │       ├── logs/
│   │       │   └── tensorboard/
│   │       ├── metrics.csv
│   │       ├── heatmaps/           # 热力图 PNG / npy 每张的命名包含样本 id
│   │       ├── reconstructions/    # 重建样本（图像网格）
│   │       └── attack_logs.json
│   └── summary/            # 实验汇总、表格、图
└── tests/                  # 单元测试 / 集成测试（选做）
    └── test_dataset.py

✅ 快速验证环境是否正常

在容器中运行：

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"


如果输出：

CUDA available: True


说明 GPU 加速环境安装成功 🎉"# reconstruction-attack-for-federated-class-unlearning" 
