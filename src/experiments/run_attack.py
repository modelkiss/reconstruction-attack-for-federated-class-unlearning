from src.attack.label_inference import confidence_label_inference
from src.models.model_utils import get_model
import torch

# 假设你已经知道这些：
dataset_name = "CIFAR10"
model_name = "resnet20"
device = "cuda"

# 1) 准备测试集 DataLoader（用你项目的数据工厂或 torchvision）
from torchvision import datasets, transforms
test_tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
test_set = datasets.CIFAR10(root="data/processed/CIFAR10", train=False, download=True, transform=test_tf)
from torch.utils.data import DataLoader
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

# 2) 构建模型与载入 before/after 权重
model_before = get_model(model_name, dataset_name, device=device)
model_after  = get_model(model_name, dataset_name, device=device)
state_before = torch.load("outputs/experiments/.../checkpoints/model_before.pth", map_location="cpu")["state_dict"]
state_after  = torch.load("outputs/experiments/.../checkpoints/model_after.pth", map_location="cpu")["state_dict"]
model_before.load_state_dict(state_before)
model_after.load_state_dict(state_after)

# 3) 推理
res = confidence_label_inference(
    model_before=model_before,
    model_after=model_after,
    test_loader=test_loader,
    device=device,
    top_k=3,
    save_dir="outputs/experiments/.../attack_logs"  # 可选
)
print(res["predicted_forgotten"], res["predicted_forgotten_names"])
