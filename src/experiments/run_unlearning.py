# src/experiments/run_unlearning.py
from __future__ import annotations
import argparse
import copy
import glob
import os
import sys
import time
from typing import Dict, Any, Optional, List

# 将项目根目录加入 sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, TensorDataset

# 项目内部依赖
from src.federated.server import Server
from src.utils.logging import setup_logging
from src.utils.seeds import set_seed
from src.utils.io import ensure_dir, save_json, save_checkpoint
from src.models.model_utils import get_model

# data factory（可选）
try:
    from src.data.dataset_factory import get_dataset
except Exception:
    get_dataset = None  # 允许缺省


# ---------------- YAML helpers ----------------
def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------------- JSON 序列化清洗 ----------------
def _to_jsonable(o):
    import numpy as np
    if isinstance(o, torch.Tensor):
        if o.dim() == 0:
            return o.item()
        return o.detach().cpu().tolist()
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return [_to_jsonable(x) for x in o]
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    try:
        import json
        _ = json.dumps(o)  # type: ignore
        return o
    except Exception:
        return str(o)


# ---------------- checkpoint 查找与加载 ----------------
def _find_latest_ckpt(server_outdir: str) -> Optional[str]:
    preferred = [
        os.path.join(server_outdir, "checkpoints", "global_final.pth"),
        os.path.join(server_outdir, "checkpoints", "global_model_final.pth"),
        os.path.join(server_outdir, "checkpoints", "model_final.pth"),
        os.path.join(server_outdir, "checkpoints", "model_before.pth"),
    ]
    for p in preferred:
        if os.path.exists(p):
            return p
    cands = glob.glob(os.path.join(server_outdir, "**", "*.pth"), recursive=True)
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def _load_state_dict_from_ckpt(ckpt_path: str) -> Optional[Dict[str, torch.Tensor]]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    return None


# ---------------- 评估 ----------------
@torch.no_grad()
def _evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    ce = torch.nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        loss = ce(out, yb)
        total_loss += loss.item() * xb.size(0)
        total_correct += (out.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return {"loss": total_loss / max(1, total), "acc": total_correct / max(1, total)}


# ---------------- processed 数据优先的 DataLoader ----------------
def _tv_mean_std(dataset_name: str):
    name = dataset_name.lower()
    if "cifar10" in name or "cifar-10" in name:
        return ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    if "cifar100" in name or "cifar-100" in name:
        return ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    # MNIST / FashionMNIST
    return ((0.5,), (0.5,))

def _normalize_tensor_images(imgs: torch.Tensor, dataset_name: str) -> torch.Tensor:
    imgs = imgs.float()  # -> float32
    if imgs.max() > 1.5:
        imgs = imgs / 255.0
    mean, std = _tv_mean_std(dataset_name)
    mean_t = torch.tensor(mean, dtype=imgs.dtype).view(-1, 1, 1)
    std_t  = torch.tensor(std, dtype=imgs.dtype).view(-1, 1, 1)
    C = imgs.shape[1]
    # 灰度→3通道（适配 ResNet）
    if C == 1 and len(mean) == 3:
        imgs = imgs.repeat(1, 3, 1, 1)
        C = 3
    if C == len(mean):
        imgs = (imgs - mean_t) / std_t
    return imgs

def _load_processed_split(processed_root: str, dataset_name: str, split: str) -> Optional[TensorDataset]:
    p = os.path.join(processed_root, dataset_name, f"{split}.pt")
    if not os.path.exists(p):
        return None
    obj = torch.load(p, map_location="cpu")  # 期望 {"images": Tensor[N,C,H,W], "labels": Tensor[N]}
    imgs = obj.get("images", None)
    labs = obj.get("labels", None)
    if imgs is None or labs is None:
        return None
    imgs = _normalize_tensor_images(imgs, dataset_name)
    labs = labs.long()
    return TensorDataset(imgs, labs)

def _build_val_loader_processed(cfg: dict, processed_root: str) -> Optional[DataLoader]:
    ds_name = cfg.get("dataset", {}).get("name", "CIFAR10")
    bs_eval = cfg.get("dataset", {}).get("eval_batch_size", 256)
    nw = cfg.get("dataset", {}).get("num_workers", 4)
    dset = _load_processed_split(processed_root, ds_name, "val")
    if dset is None:
        return None
    return DataLoader(dset, batch_size=bs_eval, shuffle=False, num_workers=nw, pin_memory=True)

def _build_public_loader_processed(cfg: dict, processed_root: str, forget_classes: List[int]) -> Optional[DataLoader]:
    ds_name = cfg.get("dataset", {}).get("name", "CIFAR10")
    bs_tr = cfg.get("federated", {}).get("client_batch_size", 64)
    nw = cfg.get("dataset", {}).get("num_workers", 4)
    dset = _load_processed_split(processed_root, ds_name, "train")
    if dset is None:
        return None
    if forget_classes:
        keep_idx = []
        labs = dset.tensors[1]
        for i in range(len(labs)):
            if int(labs[i].item()) not in forget_classes:
                keep_idx.append(i)
        if keep_idx:
            imgs = dset.tensors[0][keep_idx]
            labs = dset.tensors[1][keep_idx]
            dset = TensorDataset(imgs, labs)
    return DataLoader(dset, batch_size=bs_tr, shuffle=True, num_workers=nw, pin_memory=True)


# ---------------- 工厂 / torchvision 兜底（仅当 processed 缺失时） ----------------
def _build_val_loader(cfg: dict, processed_root: str) -> Optional[DataLoader]:
    ld = _build_val_loader_processed(cfg, processed_root)
    if ld is not None:
        return ld
    # 项目工厂
    if get_dataset is not None:
        try:
            ds = get_dataset(cfg.get("dataset", {}))
            if isinstance(ds, dict) and ds.get("val") is not None:
                bs_eval = cfg.get("dataset", {}).get("eval_batch_size", 256)
                nw = cfg.get("dataset", {}).get("num_workers", 4)
                return DataLoader(ds["val"], batch_size=bs_eval, shuffle=False, num_workers=nw, pin_memory=True)
        except Exception:
            pass
    # torchvision（很少会触发）
    try:
        from torchvision import datasets, transforms as T
        name = cfg.get("dataset", {}).get("name", "CIFAR10")
        root = cfg.get("dataset", {}).get("root") or os.path.join("data", "processed", name)
        mean, std = _tv_mean_std(name)
        tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        if "cifar100" in name.lower():
            tv = datasets.CIFAR100(root=root, train=False, download=True, transform=tf)
        elif "cifar10" in name.lower():
            tv = datasets.CIFAR10(root=root, train=False, download=True, transform=tf)
        elif "fashion" in name.lower():
            tv = datasets.FashionMNIST(root=root, train=False, download=True, transform=tf)
        else:
            tv = datasets.MNIST(root=root, train=False, download=True, transform=tf)
        return DataLoader(tv, batch_size=cfg.get("dataset", {}).get("eval_batch_size", 256),
                          shuffle=False, num_workers=cfg.get("dataset", {}).get("num_workers", 4), pin_memory=True)
    except Exception:
        return None

def _build_public_loader(cfg: dict, processed_root: str, forget_classes: List[int]) -> Optional[DataLoader]:
    ld = _build_public_loader_processed(cfg, processed_root, forget_classes)
    if ld is not None:
        return ld
    # 项目工厂
    if get_dataset is not None:
        try:
            ds = get_dataset(cfg.get("dataset", {}))
            if isinstance(ds, dict) and ds.get("train") is not None:
                d = ds["train"]
                if forget_classes:
                    from torch.utils.data import Subset
                    keep_idx = []
                    for i in range(len(d)):
                        _, y = d[i]
                        yv = int(y) if not isinstance(y, (list, tuple)) else int(y[0])
                        if yv not in forget_classes:
                            keep_idx.append(i)
                    if keep_idx:
                        d = Subset(d, keep_idx)
                return DataLoader(d, batch_size=cfg.get("federated", {}).get("client_batch_size", 64),
                                  shuffle=True, num_workers=cfg.get("dataset", {}).get("num_workers", 4), pin_memory=True)
        except Exception:
            pass
    # torchvision（很少会触发）
    try:
        from torchvision import datasets, transforms as T
        name = cfg.get("dataset", {}).get("name", "CIFAR10")
        root = cfg.get("dataset", {}).get("root") or os.path.join("data", "processed", name)
        mean, std = _tv_mean_std(name)
        tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)]) \
             if "cifar" in name.lower() else T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        if "cifar100" in name.lower():
            tv = datasets.CIFAR100(root=root, train=True, download=True, transform=tf)
        elif "cifar10" in name.lower():
            tv = datasets.CIFAR10(root=root, train=True, download=True, transform=tf)
        elif "fashion" in name.lower():
            tv = datasets.FashionMNIST(root=root, train=True, download=True, transform=tf)
        else:
            tv = datasets.MNIST(root=root, train=True, download=True, transform=tf)
        # 过滤遗忘类
        if forget_classes:
            from torch.utils.data import Subset
            keep_idx = []
            for i in range(len(tv)):
                _, y = tv[i]
                if int(y) not in forget_classes:
                    keep_idx.append(i)
            if keep_idx:
                tv = Subset(tv, keep_idx)
        return DataLoader(tv, batch_size=cfg.get("federated", {}).get("client_batch_size", 64),
                          shuffle=True, num_workers=cfg.get("dataset", {}).get("num_workers", 4), pin_memory=True)
    except Exception:
        return None


# ---------------- 可选热力图 ----------------
def _maybe_generate_saliency(logger, phase: str, model, cfg: Dict[str, Any], out_dir: str, processed_root: str):
    if not cfg.get("experiment", {}).get("save_heatmaps", False):
        return
    try:
        from src.explainability.saliency import generate_saliency
    except Exception as e:
        logger.warning(f"[saliency] 未找到 saliency 模块，跳过热力图生成：{e}")
        return

    loader = _build_val_loader(cfg, processed_root)
    if loader is None:
        logger.warning("[saliency] 无法构建 val loader，跳过热力图生成")
        return

    device = cfg.get("experiment", {}).get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        res = generate_saliency(
            model=model,
            dataloader=loader,
            device=torch.device(device),
            method=cfg.get("experiment", {}).get("heatmap", {}).get("method", "grad"),
            max_batches=int(cfg.get("experiment", {}).get("heatmap", {}).get("max_batches", 1)),
        )
        save_to = os.path.join(out_dir, "heatmaps")
        ensure_dir(save_to)
        import numpy as np
        np.save(os.path.join(save_to, f"{phase}_images.npy"), res.get("images"))
        np.save(os.path.join(save_to, f"{phase}_heatmaps.npy"), res.get("heatmaps"))
        np.save(os.path.join(save_to, f"{phase}_labels.npy"), res.get("labels"))
        logger.info(f"[saliency] {phase} 热力图已保存到 {save_to}")
    except Exception as e:
        logger.warning(f"[saliency] 生成/保存热力图失败：{e}")


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Run federated unlearning (handoff from run_federated)")
    p.add_argument("--config", "-c", type=str, required=False,
                   help="实验配置 YAML（若提供 --server-outdir，则以其中 config.yaml 为主）")
    p.add_argument("--unlearn-config", type=str, default="configs/unlearning_default.yaml",
                   help="unlearning 默认配置，合并进最终配置")
    p.add_argument("--server-outdir", type=str, default=None,
                   help="指向 run_federated 输出目录（含 config.yaml、checkpoints/ 等）")
    p.add_argument("--device", type=str, default=None, help="覆盖设备（cpu/cuda），默认按配置或自动检测")
    p.add_argument("--processed-root", type=str, default="data/processed", help="已预处理数据根目录（优先使用）")
    p.add_argument("--output-root", type=str, default="outputs/experiments", help="本次 unlearning 的输出根目录")
    return p.parse_args()

def detect_device(preferred: Optional[str]) -> str:
    if preferred:
        preferred = preferred.lower()
        if preferred in ("cpu", "cuda"):
            if preferred == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA 不可用，回退为 CPU")
                return "cpu"
            return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- 主流程 ----------------
import json

def main():
    args = parse_args()

    # 读取配置
    base_cfg: Dict[str, Any] = {}
    server_cfg_path = os.path.join(args.server_outdir, "config.yaml") if args.server_outdir else None
    if server_cfg_path and os.path.exists(server_cfg_path):
        base_cfg = load_yaml(server_cfg_path)
    elif args.config:
        base_cfg = load_yaml(args.config)
    else:
        raise ValueError("必须提供 --server-outdir（含 config.yaml）或 --config 其一。")

    # 合并 unlearning 默认配置
    ul_cfg_path = args.unlearn_config if args.unlearn_config and os.path.exists(args.unlearn_config) else None
    if ul_cfg_path:
        base_cfg = deep_merge(base_cfg, {"unlearning": load_yaml(ul_cfg_path).get("unlearning", {})})

    # 设备 / 种子
    device = detect_device(args.device or base_cfg.get("experiment", {}).get("device"))
    seed = base_cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    print(f"✅ Random seed set to {seed}")

    # 输出目录与日志
    tag = f"unlearn_{time.strftime('%Y%m%d_%H%M%S')}"
    out_root = os.path.join(args.output_root, tag)
    ensure_dir(out_root)
    logger = setup_logging(out_root, log_name="run_unlearning")
    logger.info(f"[run_unlearning] 输出目录：{out_root}")

    # 数据/模型名
    dataset_name = base_cfg.get("dataset", {}).get("name", "CIFAR10")
    model_name = base_cfg.get("model", {}).get("backbone", "resnet20")

    # 获取 model_before：优先复用 server_outdir 的 checkpoint
    state_before: Optional[Dict[str, torch.Tensor]] = None
    server_out_used: Optional[str] = None
    if args.server_outdir and os.path.isdir(args.server_outdir):
        ckpt = _find_latest_ckpt(args.server_outdir)
        if ckpt:
            sd = _load_state_dict_from_ckpt(ckpt)
            if sd:
                state_before = sd
                server_out_used = args.server_outdir
                logger.info(f"[handoff] 复用 {args.server_outdir} 的 checkpoint：{ckpt}")
            else:
                logger.warning(f"[handoff] 找到 {ckpt} 但无法解析 state_dict，准备回退到重训")

    if state_before is None:
        logger.info("[handoff] 未提供可用 checkpoint，开始重新联邦训练以获得 model_before")
        server = Server(base_cfg, processed_root=args.processed_root, output_dir=args.output_root, device=device)
        _ = server.run()
        state_before = server.global_state
        server_out_used = server.out_dir
        logger.info(f"[handoff] 重训完成，来源：{server_out_used}")

    # 构建 model_before 并保存快照
    model_before = get_model(model_name, dataset_name, device=device)
    model_before.load_state_dict(state_before)

    ckpt_dir = os.path.join(out_root, "checkpoints")
    ensure_dir(ckpt_dir)
    before_path = os.path.join(ckpt_dir, "model_before.pth")
    save_checkpoint({"state_dict": state_before, "meta": {"phase": "before", "from": server_out_used}}, before_path)
    logger.info(f"[ckpt] 保存 model_before 至 {before_path}")

    # BEFORE 评估（processed 优先）
    val_loader = _build_val_loader(base_cfg, args.processed_root)
    if val_loader is not None:
        m_before = _evaluate(model_before, val_loader, device)
        save_json({"phase": "before", **m_before}, os.path.join(out_root, "metrics_before.json"))
        logger.info(f"[eval] BEFORE  val/loss={m_before['loss']:.4f}, val/acc={m_before['acc']:.4f}")
    else:
        logger.warning("[eval] 未构建到 val loader，跳过 BEFORE 评估")

    # BEFORE 热力图（可选）
    _maybe_generate_saliency(logger, "before", model_before, base_cfg, out_dir=out_root, processed_root=args.processed_root)

    # 执行遗忘
    ul = base_cfg.get("unlearning", {})
    method = str(ul.get("method", "fucp")).lower()
    target_classes = list(ul.get("target_classes", []))
    logger.info(f"[unlearning] method={method}, target_classes={target_classes}")

    if method == "retrain":
        from src.unlearning.retrain_unlearner import RetrainUnlearner
        retr = RetrainUnlearner(cfg=copy.deepcopy(base_cfg), forgotten_classes=target_classes, device=device,
                                output_dir=os.path.join(out_root, "unlearning_retrain"))
        result = retr.run_unlearning()
        state_after = result["model_after"]
        model_after = get_model(model_name, dataset_name, device=device)
        model_after.load_state_dict(state_after)
        result_json = {k: v for k, v in result.items() if k != "model_after"}
    else:
        # 近似类方法需要 public_dataloader（优先 processed/train.pt）
        public_loader = _build_public_loader(base_cfg, args.processed_root, forget_classes=target_classes)
        if public_loader is None:
            raise RuntimeError(
                "无法构建 public_dataloader（近似遗忘需要公共数据）。\n"
                "请检查 data/processed/<DATASET>/train.pt 是否存在，或改用 retrain 方法。"
            )
        from src.unlearning.approximate_unlearner import ApproximateUnlearner
        approx = ApproximateUnlearner(
            model=model_before,
            forgotten_classes=target_classes,
            public_dataloader=public_loader,
            device=device,
            output_dir=os.path.join(out_root, f"unlearning_{method}"),
            method=method,
            # 将 unlearning_default.yaml 的相关超参映射到 ApproximateUnlearner
            lr=float(ul.get("fucp", {}).get("fine_tune_lr", 1e-4) if method == "fucp" else ul.get(method, {}).get("lr", 1e-4)),
            steps=int(ul.get(method, {}).get("max_steps", 500)),
            finetune_steps=int(ul.get("fucp", {}).get("fine_tune_epochs", 0) * 100),  # epoch->steps 粗略映射
            prune_fraction=float(ul.get("fucp", {}).get("pruning_rate", 0.01)),
            prune_mode="zero",
            importance_estimator="grad",
            projector_dim=int(ul.get("fucrt", {}).get("projection_dim", 128)),
            projector_lr=float(ul.get("fucrt", {}).get("lr", 1e-3)),
            projector_steps=int(ul.get("fucrt", {}).get("transform_epochs", 5) * 200),
            adv_lr=float(ul.get("fedau", {}).get("lr", 1e-3)),
            adv_steps=int(ul.get("fedau", {}).get("max_steps", 500)),
            adv_weight_nonforgot=float(ul.get("fedau", {}).get("adv_lambda", 0.1)),
        )
        result = approx.run_unlearning()
        state_after = result["model_after"]
        model_after = approx.model
        result_json = {k: v for k, v in result.items() if k != "model_after"}

    # 保存 AFTER（权重->pth；结果->json）
    after_path = os.path.join(ckpt_dir, "model_after.pth")
    save_checkpoint({"state_dict": state_after, "meta": {"phase": "after", "method": method}}, after_path)
    result_json = _to_jsonable(result_json)
    save_json(result_json, os.path.join(out_root, "unlearning_result.json"))
    logger.info(f"[ckpt] 保存 model_after 至 {after_path}")

    # AFTER 评估
    if val_loader is not None:
        m_after = _evaluate(model_after, val_loader, device)
        save_json({"phase": "after", "method": method, **m_after}, os.path.join(out_root, "metrics_after.json"))
        logger.info(f"[eval] AFTER   val/loss={m_after['loss']:.4f}, val/acc={m_after['acc']:.4f}")
    else:
        logger.warning("[eval] 未构建到 val loader，跳过 AFTER 评估")

    # AFTER 热力图（可选）
    _maybe_generate_saliency(logger, "after", model_after, base_cfg, out_dir=out_root, processed_root=args.processed_root)

    # 汇总
    summary = {
        "server_out_used": args.server_outdir,
        "unlearning_out_dir": out_root,
        "method": method,
        "target_classes": target_classes,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(summary, os.path.join(out_root, "summary.json"))
    print(f"[DONE] Unlearning 完成。输出目录：{out_root}")

if __name__ == "__main__":
    main()
