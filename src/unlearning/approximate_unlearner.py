"""
approximate_unlearner.py

Three representative approximate federated unlearning strategies (implemented as methods):
  - FUCP   : "Federated Unlearning by Class Pruning" (pruning-based parameter editing)
  - FUCRT  : "Federated Unlearning via Representation Transformation" (feature projector)
  - FedAU  : "Federated Adversarial Untraining" (targeted anti-training on public data)

Design goals:
  - Each method aims to efficiently remove (or strongly reduce) model's capability regarding
    certain forgotten_classes without full retraining.
  - Methods accept a public_dataloader (labeled or unlabeled, depending on method).
  - Methods log basic stats and return a dict:
      {"model_after": state_dict, "unlearned_classes": [...], "stats": {...}}
  - Methods save a checkpoint of the model after modification via BaseUnlearner.save_states()

Notes on method choice:
  - FUCP (pruning) directly edits parameters deemed important to forgotten classes.
    Good for quick, localized edits. Needs careful importance estimation and usually a small fine-tune.
  - FUCRT (representation conversion) learns a light-weight transform (projector) that
    removes the class-specific directions from intermediate features (more "soft" and reversible).
    Often preserves performance on non-forgotten classes.
  - FedAU (adversarial untraining) is an advanced anti-training: actively *maximizes* model's
    loss on forgotten-class examples (or pseudo-labeled public examples), while optionally
    preserving non-forgotten behavior via auxiliary regularization.
"""

from __future__ import annotations
import os
import copy
import math
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.unlearning.base_unlearner import BaseUnlearner
from src.utils.logging import setup_logging
from src.utils.io import save_checkpoint


# helper utilities
def _to_device_state_dict(sd: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in sd.items()}


def _clone_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


class ApproximateUnlearner(BaseUnlearner):
    """
    Unified approximate/unlearning wrapper with three methods:
      - method == "fucp"  -> FUCP (pruning-based)
      - method == "fucrt" -> FUCRT (representation conversion / projector)
      - method == "fedau" -> FedAU (adversarial/untraining via gradient ascent on forgotten classes)
    """
    def __init__(
        self,
        model: nn.Module,
        forgotten_classes: List[int],
        public_dataloader,
        device: str = "cpu",
        output_dir: str = "./outputs/unlearning/approximate",
        method: str = "fucp",  # "fucp" | "fucrt" | "fedau"
        # common hyperparams
        lr: float = 1e-4,
        steps: int = 500,
        finetune_steps: int = 200,
        # FUCP params
        prune_fraction: float = 0.01,
        prune_mode: str = "zero",  # "zero" | "reinit"
        importance_estimator: str = "grad",  # "grad" | "fisher"
        # FUCRT params
        projector_dim: int = 128,
        projector_lr: float = 1e-3,
        projector_steps: int = 1000,
        # FedAU params
        adv_lr: float = 1e-3,
        adv_steps: int = 500,
        adv_weight_nonforgot: float = 0.1,
    ):
        super().__init__(model, dataset_name=getattr(model, "__dataset__", "unknown"), device=device, output_dir=output_dir)
        self.model = model
        self.forgotten_classes = sorted(list(set(forgotten_classes)))
        self.public_dataloader = public_dataloader
        self.device = torch.device(device)
        self.method = method.lower()
        self.lr = lr
        self.steps = int(steps)
        self.finetune_steps = int(finetune_steps)
        # FUCP
        self.prune_fraction = float(prune_fraction)
        self.prune_mode = prune_mode
        self.importance_estimator = importance_estimator
        # FUCRT
        self.projector_dim = int(projector_dim)
        self.projector_lr = projector_lr
        self.projector_steps = int(projector_steps)
        # FedAU
        self.adv_lr = adv_lr
        self.adv_steps = int(adv_steps)
        self.adv_weight_nonforgot = adv_weight_nonforgot

        self.logger = setup_logging(self.output_dir, log_name=f"approx_unlearner_{self.method}")
        self.model.to(self.device)

    def run_unlearning(self) -> Dict[str, Any]:
        self.logger.info(f"Approximate unlearning starting: method={self.method}, forgotten_classes={self.forgotten_classes}")

        if self.method == "fucp":
            stats = self._run_fucp()
        elif self.method == "fucrt":
            stats = self._run_fucrt()
        elif self.method == "fedau":
            stats = self._run_fedau()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        model_after = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        path = self.save_states(model_after, f"after_{self.method}")
        stats.update({"model_path": path})
        self.logger.info(f"Done approximate unlearning ({self.method}). Model saved at {path}")
        return {"model_after": model_after, "unlearned_classes": self.forgotten_classes, "stats": stats}

    # -----------------------------
    # common importance estimator
    # -----------------------------
    def _estimate_importance(self, data_iter, estimator="grad", max_samples: Optional[int] = 1000) -> Dict[str, torch.Tensor]:
        """
        Estimate per-parameter importance for forgotten classes.
        - 'grad': accumulate squared gradients of loss w.r.t parameters on forgotten-class samples
        - 'fisher': approximate fisher diagonal (expected squared grad of log-likelihood)
        Returns dict param_name -> importance tensor (on CPU)
        """
        self.model.eval()
        importance = {k: torch.zeros_like(v.detach().cpu()) for k, v in self.model.state_dict().items()}
        device = self.device
        count = 0
        criterion = nn.CrossEntropyLoss(reduction="sum")

        for xb, yb in data_iter:
            if max_samples and count >= max_samples:
                break
            xb = xb.to(device)
            yb = yb.to(device)
            # mask forgotten classes
            mask = torch.zeros_like(yb, dtype=torch.bool)
            for c in self.forgotten_classes:
                mask |= (yb == c)
            if not mask.any():
                continue
            xb_sel = xb[mask]
            yb_sel = yb[mask]
            if xb_sel.size(0) == 0:
                continue

            self.model.zero_grad()
            out = self.model(xb_sel)
            loss = criterion(out, yb_sel)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                importance[name] += (param.grad.detach().cpu() ** 2)
            count += xb_sel.size(0)

        if count > 0:
            for k in importance:
                importance[k] = importance[k] / float(max(1, count))
        else:
            # fallback: magnitude proxy
            self.logger.warning("No forgotten-class samples during importance estimation; fallback to weight magnitude proxy.")
            for k, v in self.model.state_dict().items():
                importance[k] = v.detach().cpu().abs()

        return importance

    # -----------------------------
    # FUCP: pruning-based parameter editing
    # -----------------------------
    def _run_fucp(self) -> Dict[str, Any]:
        """
        FUCP (pruning-based):
          1) Estimate importance of parameters for forgotten classes (grad or fisher)
          2) Choose top-k important parameters (by prune_fraction) and prune them:
             - mode "zero": set to 0
             - mode "reinit": reinitialize (Kaiming for weights, zero for biases)
          3) Optionally fine-tune remaining model on public data (to recover non-forgotten performance)
        """
        self.logger.info("Starting FUCP (pruning-based) unlearning")
        # Step 1: importance
        importance = self._estimate_importance(self.public_dataloader, estimator=self.importance_estimator, max_samples=500)
        flat = []
        for name, imp in importance.items():
            flat.append((name, imp.view(-1)))
        all_vals = torch.cat([v for _, v in flat])
        total_params = all_vals.numel()
        k = max(1, int(self.prune_fraction * total_params))
        if k >= total_params:
            thresh = -float("inf")
        else:
            # pick threshold to select top-k
            thresh = torch.kthvalue(all_vals, all_vals.numel() - k + 1).values.item()
        self.logger.info(f"FUCP: total_params={total_params}, prune_k={k}, threshold={thresh:.6e}")

        # Step 2: apply pruning
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                imp = importance.get(name)
                if imp is None:
                    continue
                # create mask of elements to prune
                mask = (imp.to(param.device) >= thresh).view(param.data.size())
                if mask.any():
                    if self.prune_mode == "zero":
                        param.data[mask] = 0.0
                    elif self.prune_mode == "reinit":
                        if param.dim() >= 2:
                            nn.init.kaiming_normal_(param.data)
                        else:
                            param.data.zero_()
                    else:
                        raise ValueError(f"Unknown prune_mode: {self.prune_mode}")

        # Optional fine-tune to recover non-forgotten functionality
        ft_stats = {}
        if self.finetune_steps > 0 and self.public_dataloader is not None:
            self.logger.info(f"FUCP: fine-tuning for {self.finetune_steps} steps on public data (lr={self.lr})")
            self.model.train()
            opt = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            total = 0
            losses = []
            for epoch in range(max(1, math.ceil(self.finetune_steps))):
                for xb, yb in self.public_dataloader:
                    if total >= self.finetune_steps:
                        break
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    opt.zero_grad()
                    out = self.model(xb)
                    # training with full labels may reintroduce forgotten info if public data contains forgotten class.
                    # Option: mask out forgotten-class loss (we minimize only for non-forgotten classes)
                    mask = torch.ones_like(yb, dtype=torch.bool)
                    for c in self.forgotten_classes:
                        mask &= (yb != c)
                    if mask.any():
                        loss = criterion(out[mask], yb[mask])
                        loss.backward()
                        opt.step()
                        losses.append(loss.item())
                    total += 1
                if total >= self.finetune_steps:
                    break
            ft_stats["finetune_steps"] = total
            ft_stats["finetune_avg_loss"] = float(np.mean(losses) if losses else 0.0)
            self.logger.info(f"FUCP fine-tune done. steps={total}, avg_loss={ft_stats['finetune_avg_loss']:.6f}")

        return {"method": "fucp", "prune_fraction": self.prune_fraction, "prune_mode": self.prune_mode, **ft_stats}

    # -----------------------------
    # FUCRT: representation conversion (projection) method
    # -----------------------------
    def _run_fucrt(self) -> Dict[str, Any]:
        """
        FUCRT (representation conversion):
          - Attach a lightweight linear projector P to an intermediate layer (penultimate features).
          - Train P such that the classifier's logits for forgotten classes are suppressed,
            while preserving logits for other classes (via a distillation/regression term).
          - Implementation approach:
              * Hook to extract penultimate features (we look for a tensor named 'features' via forward hook,
                otherwise attempt to use the output of the last conv block by registering on a provided layer).
              * Projector is initialized as identity-like mapping and trained on public data.
        Notes:
          - This is a 'soft' modification: original model weights are kept; we apply projector in forward path.
          - After training, we merge the projector into model by folding (if possible) or keep as wrapper.
        """
        self.logger.info("Starting FUCRT (representation conversion) unlearning")

        # find a feature extraction point: try to locate a module named 'avgpool' or last conv
        # We implement a flexible hook: capture penultimate feature (flattened) during forward
        features = {}
        hook_handles = []

        # attempt to hook at the common attribute 'avgpool' or the last conv layer
        target_module = None
        for name, module in self.model.named_modules():
            if name.endswith("avgpool") or name.endswith("pool") or "layer4" in name:
                target_module = module
        if target_module is None:
            # fallback to the model itself (last activation)
            target_module = list(self.model.modules())[-1]

        def forward_hook(module, inp, out):
            # store a flattened representation
            # out may be (B, C, 1, 1) or (B, C, H, W)
            feat = out
            if feat.dim() > 2:
                feat = torch.flatten(feat, 1)
            features["feat"] = feat.detach()

        handle = target_module.register_forward_hook(forward_hook)
        hook_handles.append(handle)

        # determine feature dim by a single forward pass using a sample from dataloader
        sample_x = None
        for batch in self.public_dataloader:
            if isinstance(batch, (list, tuple)):
                sample_x = batch[0]
            else:
                sample_x = batch
            break
        if sample_x is None:
            self.logger.error("FUCRT: public_dataloader is empty; aborting FUCRT.")
            for h in hook_handles:
                h.remove()
            return {"method": "fucrt", "error": "empty_public_dataloader"}

        sample_x = sample_x.to(self.device)
        # run one forward to get feature dim
        self.model.eval()
        with torch.no_grad():
            _ = self.model(sample_x[0:1] if sample_x.dim() == 4 else sample_x)

        if "feat" not in features:
            for h in hook_handles:
                h.remove()
            self.logger.error("FUCRT: failed to capture features from model; aborting.")
            return {"method": "fucrt", "error": "no_features_hooked"}

        feat_dim = features["feat"].shape[1]
        self.logger.info(f"FUCRT: detected feature dim = {feat_dim}")

        # build projector: linear layer mapping feat_dim -> feat_dim (initialized near identity)
        P = nn.Linear(feat_dim, feat_dim, bias=False).to(self.device)
        # init as identity (approx) if square
        with torch.no_grad():
            eye = torch.eye(feat_dim, device=self.device)
            if P.weight.shape == eye.shape:
                P.weight.data.copy_(eye * 1.0)
            else:
                nn.init.kaiming_normal_(P.weight)

        # We'll create a small wrapper that uses the projector in forward pass: we implement training by
        # running the model forward, capturing features, replacing with P(feat) before classifier logits.
        # To do this without rewriting model internals, we perform manual forward: run submodule up to hook,
        # then apply projector, then run remaining classifier layers. For complexity reasons, here we adopt a
        # pragmatic approach: we approximate by training P to minimize teacher logits for forgotten classes
        # when fed the captured features, and to preserve other logits via MSE to teacher.
        # So we need a reference mapping feat -> logits. We'll build a small 'head' that maps feat -> logits by
        # forward passing full model and capturing mapping via least squares (linear probe), then use P on feat.

        # Step A: build a simple linear head that maps features to logits (train a probe)
        # Collect a moderate number of (feat, logits) pairs from public data
        self.model.eval()
        probe_X = []
        probe_Y = []
        num_probe = 0
        max_probe = 256
        softmax = nn.Softmax(dim=1)
        for batch in self.public_dataloader:
            if num_probe >= max_probe:
                break
            if isinstance(batch, (list, tuple)):
                xb = batch[0]
            else:
                xb = batch
            xb = xb.to(self.device)
            with torch.no_grad():
                out = self.model(xb)
                # trigger hook to populate features (hook saved features of last sample; to be safe, run per sample)
            # forward_hook stored batched features in features["feat"]
            if "feat" not in features:
                continue
            feat = features["feat"].detach().cpu()
            logits = out.detach().cpu()
            probe_X.append(feat)
            probe_Y.append(logits)
            num_probe += feat.shape[0]
        if len(probe_X) == 0:
            for h in hook_handles:
                h.remove()
            self.logger.error("FUCRT: failed to collect probe data; aborting.")
            return {"method": "fucrt", "error": "no_probe_data"}

        probe_X = torch.cat(probe_X, dim=0)
        probe_Y = torch.cat(probe_Y, dim=0)
        # train linear probe: feat_dim -> num_classes
        num_classes = probe_Y.shape[1]
        probe_head = nn.Linear(feat_dim, num_classes).to(self.device)
        probe_opt = optim.Adam(probe_head.parameters(), lr=1e-3)
        probe_crit = nn.MSELoss()
        probe_X_t = probe_X.to(self.device)
        probe_Y_t = probe_Y.to(self.device)
        for _ in range(50):
            probe_opt.zero_grad()
            pred = probe_head(probe_X_t)
            loss = probe_crit(pred, probe_Y_t)
            loss.backward()
            probe_opt.step()
        self.logger.info("FUCRT: trained linear probe to approximate logits from features")

        # Step B: train projector P to suppress forgotten-class logits when applied to features
        # Loss = KL( softmax(probe_head(P(feat))) , target ), where target reduces mass on forgotten classes
        optimizer = optim.Adam(P.parameters(), lr=self.projector_lr)
        kl = nn.KLDivLoss(reduction="batchmean")
        steps_done = 0
        self.model.eval()
        for epoch in range(max(1, self.projector_steps)):
            for batch in self.public_dataloader:
                if steps_done >= self.projector_steps:
                    break
                if isinstance(batch, (list, tuple)):
                    xb = batch[0]
                else:
                    xb = batch
                xb = xb.to(self.device)
                with torch.no_grad():
                    _ = self.model(xb)
                    feat = features.get("feat")
                    if feat is None:
                        continue
                    feat = feat.to(self.device)
                    teacher_logits = probe_head(feat)  # using probe as surrogate for classifier on feat
                    teacher_probs = torch.softmax(teacher_logits, dim=1)
                    # zero out forgotten classes mass and renormalize
                    teacher_probs[:, self.forgotten_classes] = 0.0
                    s = teacher_probs.sum(dim=1, keepdim=True)
                    s[s == 0] = 1.0
                    target_probs = teacher_probs / s

                # apply projector then probe_head
                proj_feat = P(feat)
                s_logits = probe_head(proj_feat)
                s_logprob = torch.log_softmax(s_logits, dim=1)
                loss_kld = kl(s_logprob, target_probs.to(self.device))
                optimizer.zero_grad()
                loss_kld.backward()
                optimizer.step()

                steps_done += 1
            if steps_done >= self.projector_steps:
                break

        # After training, integrate projector:
        # Because we used a probe (approximate head), perfect folding into original classifier is non-trivial.
        # A practical option: keep projector separately and create a wrapper that applies P to features at inference.
        # For now, we'll store projector weights and provide a "note" in stats that the projector must be
        # applied in forward path for effective forgetting. Optionally you can fold P into the first linear layer
        # of the classifier if architecture permits.
        for h in hook_handles:
            h.remove()

        # Save projector to outputs for later use
        proj_path = os.path.join(self.output_dir, f"projector_{self.projector_steps}_steps.pth")
        torch.save({"state_dict": P.state_dict(), "feat_dim": feat_dim}, proj_path)
        self.logger.info(f"FUCRT: projector saved to {proj_path}")

        return {"method": "fucrt", "projector_path": proj_path, "projector_steps": steps_done}

    # -----------------------------
    # FedAU: adversarial untraining / reverse training
    # -----------------------------
    def _run_fedau(self) -> Dict[str, Any]:
        """
        FedAU (adversarial untraining):
          - For public data (labeled or pseudo-labeled), perform gradient *ascent* on loss for forgotten-class samples
            (i.e., increase model's loss on those classes), while optionally applying a regularizer to
            keep predictions for non-forgotten classes stable.
          - Implementation detail: we do small ascent steps (via optimizer that adds negative gradient).
            Equivalent to minimizing a loss = -lambda * L_forgot + reg * L_nonforgot
        """
        self.logger.info("Starting FedAU (adversarial untraining)")

        # keep a teacher copy for stability regularization
        teacher = copy.deepcopy(self.model).to(self.device).eval()

        # optimizer performs gradient ascent: we accomplish this by minimizing loss_adv = -L_forgot + reg*L_nonforgot
        optimizer = optim.SGD(self.model.parameters(), lr=self.adv_lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss(reduction="mean")
        total_steps = 0
        losses = []
        lambda_adv = 1.0
        reg_weight = float(self.adv_weight_nonforgot)

        for epoch in range(max(1, math.ceil(self.adv_steps))):
            for batch in self.public_dataloader:
                if total_steps >= self.adv_steps:
                    break
                if isinstance(batch, (list, tuple)):
                    xb, yb = batch[0], batch[1]
                else:
                    # unlabeled case: use teacher predictions as pseudo-labels
                    xb = batch
                    with torch.no_grad():
                        t_out = teacher(xb.to(self.device))
                        yb = torch.argmax(t_out, dim=1).cpu()
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                out = self.model(xb)

                # build lost for forgotten-class samples (we want to maximize it)
                mask_forgot = torch.zeros_like(yb, dtype=torch.bool)
                for c in self.forgotten_classes:
                    mask_forgot |= (yb == c)

                loss_forgot = torch.tensor(0.0, device=self.device)
                if mask_forgot.any():
                    loss_forgot = criterion(out[mask_forgot], yb[mask_forgot])

                # regularization: keep non-forgotten predictions similar to teacher (distillation)
                loss_reg = torch.tensor(0.0, device=self.device)
                mask_non = ~mask_forgot
                if mask_non.any():
                    with torch.no_grad():
                        t_logits = teacher(xb[mask_non])
                    s_logprob = torch.log_softmax(out[mask_non], dim=1)
                    t_prob = torch.softmax(t_logits, dim=1)
                    loss_reg = nn.KLDivLoss(reduction="batchmean")(s_logprob, t_prob)

                # adversarial objective: minimize (-lambda * loss_forgot + reg_weight * loss_reg)
                loss_total = (-lambda_adv * loss_forgot) + (reg_weight * loss_reg)
                loss_total.backward()
                # Gradient step will reduce loss_total; because -loss_forgot term, it performs ascent on loss_forgot.
                optimizer.step()

                losses.append(loss_total.item())
                total_steps += 1
            if total_steps >= self.adv_steps:
                break

        avg_loss = float(np.mean(losses) if losses else 0.0)
        self.logger.info(f"FedAU finished. steps={total_steps}, avg_obj={avg_loss:.6f}")
        return {"method": "fedau", "steps": total_steps, "avg_obj": avg_loss, "reg_weight": reg_weight}

