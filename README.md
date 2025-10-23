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

# 一键跑完整流程（联邦训练→遗忘→标签推理→数据重建）
python -m src.experiments.run_full_pipeline --config configs/experiments/full_pipeline_all.yaml

`full_pipeline_all.yaml` 内预置了 CIFAR-10/CIFAR-100、MNIST、Fashion-MNIST 的基线与防御场景（安全聚合、差分隐私）。
脚本会在 `outputs/pipeline/<时间戳>` 下生成每个场景的模型权重、标签推理结果、重建图像以及汇总 JSON；可使用 `--scenario` 仅运行某个场景。

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