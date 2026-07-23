import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed=42, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def print_vram_usage():
    if not torch.cuda.is_available():
        print("[VRAM] CUDA is unavailable; memory statistics were skipped.")
        return
    max_memory = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"[VRAM] Peak allocated memory: {max_memory:.2f} GB")


def prepare_mask(labels):
    """Convert masks to Bx1xHxW float tensors with values in {0, 1}."""
    if labels.dim() == 3:
        labels = labels.unsqueeze(1)
    labels = labels.float()
    if labels.max() > 1:
        labels = labels / 255.0
    return (labels > 0.5).float()


class MetricsTracker:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    @torch.no_grad()
    def update(self, preds, labels):
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)
        preds = (preds > self.threshold).bool()
        labels = (labels > 0.5).bool()

        self.TP += ((preds == 1) & (labels == 1)).sum().item()
        self.FP += ((preds == 1) & (labels == 0)).sum().item()
        self.TN += ((preds == 0) & (labels == 0)).sum().item()
        self.FN += ((preds == 0) & (labels == 1)).sum().item()

    def get_metrics(self):
        precision = self.TP / (self.TP + self.FP + 1e-8)
        recall = self.TP / (self.TP + self.FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = self.TP / (self.TP + self.FP + self.FN + 1e-8)
        oa = (self.TP + self.TN) / (
            self.TP + self.TN + self.FP + self.FN + 1e-8
        )
        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "IoU": iou,
            "OA": oa,
        }


class EADLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, boundary_weight=0.2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight

        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("laplace_kernel", kernel)

    def boundary_map(self, x):
        kernel = self.laplace_kernel.to(device=x.device, dtype=x.dtype)
        return F.conv2d(x, kernel, padding=1).abs()

    def forward(self, logits, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)

        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        focal_loss = (focal_weight * bce).mean()

        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice_loss = 1 - (2 * intersection + 1e-5) / (union + 1e-5)
        dice_loss = dice_loss.mean()

        pred_edge = self.boundary_map(probs)
        target_edge = self.boundary_map(targets)
        boundary_loss = F.l1_loss(pred_edge, target_edge)

        return focal_loss + dice_loss + self.boundary_weight * boundary_loss

