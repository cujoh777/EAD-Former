import torch
import torch.nn.functional as F


def _ensure_4d_binary(x):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(1)
    elif x.ndim != 4:
        raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")

    return (x > 0).float()


def _erode_binary(mask, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    padding = kernel_size // 2
    inverse = 1.0 - mask
    eroded = 1.0 - F.max_pool2d(
        inverse,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )
    return (eroded > 0.5).float()


def _dilate_binary(mask, radius):
    if radius <= 0:
        return mask

    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )
    return (dilated > 0.5).float()


def mask_to_boundary(mask, kernel_size=3, mode="inner"):
    mask = _ensure_4d_binary(mask)

    if mode == "symmetric":
        radius = kernel_size
        dilated = _dilate_binary(mask, radius=radius)
        eroded = _erode_binary(mask, kernel_size=2 * radius + 1)
        return ((dilated - eroded) > 0.5).float()
    if mode != "inner":
        raise ValueError(f"Unsupported boundary mode: {mode}")

    eroded = _erode_binary(mask, kernel_size=kernel_size)
    boundary = mask - eroded
    return (boundary > 0.5).float()


@torch.no_grad()
def boundary_counts(
    pred,
    target,
    tolerance=2,
    boundary_kernel_size=3,
    boundary_mode="inner",
):
    pred = _ensure_4d_binary(pred)
    target = _ensure_4d_binary(target)

    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )

    pred_boundary = mask_to_boundary(
        pred,
        kernel_size=boundary_kernel_size,
        mode=boundary_mode,
    )
    target_boundary = mask_to_boundary(
        target,
        kernel_size=boundary_kernel_size,
        mode=boundary_mode,
    )

    pred_dilated = _dilate_binary(pred_boundary, radius=tolerance)
    target_dilated = _dilate_binary(target_boundary, radius=tolerance)

    matched_pred = pred_boundary * target_dilated
    matched_target = target_boundary * pred_dilated
    intersection = (pred_boundary * target_boundary).sum()
    union = ((pred_boundary + target_boundary) > 0).float().sum()

    return {
        "bf1_tp_p": matched_pred.sum().item(),
        "bf1_tp_r": matched_target.sum().item(),
        "pred_boundary": pred_boundary.sum().item(),
        "target_boundary": target_boundary.sum().item(),
        "biou_inter": intersection.item(),
        "biou_union": union.item(),
    }


def compute_boundary_metrics_from_counts(counts, eps=1e-7):
    precision = counts["bf1_tp_p"] / (counts["pred_boundary"] + eps)
    recall = counts["bf1_tp_r"] / (counts["target_boundary"] + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = counts["biou_inter"] / (counts["biou_union"] + eps)

    return {
        "B-Precision": precision,
        "B-Recall": recall,
        "B-F1": f1,
        "B-IoU": iou,
    }


class BoundaryMetricTracker:
    def __init__(
        self,
        tolerance=2,
        boundary_kernel_size=3,
        boundary_mode="inner",
    ):
        self.tolerance = tolerance
        self.boundary_kernel_size = boundary_kernel_size
        self.boundary_mode = boundary_mode
        self.reset()

    def reset(self):
        self.counts = {
            "bf1_tp_p": 0.0,
            "bf1_tp_r": 0.0,
            "pred_boundary": 0.0,
            "target_boundary": 0.0,
            "biou_inter": 0.0,
            "biou_union": 0.0,
        }

    @torch.no_grad()
    def update(self, pred, target):
        batch_counts = boundary_counts(
            pred=pred,
            target=target,
            tolerance=self.tolerance,
            boundary_kernel_size=self.boundary_kernel_size,
            boundary_mode=self.boundary_mode,
        )
        for key, value in batch_counts.items():
            self.counts[key] += float(value)

    def compute(self):
        return compute_boundary_metrics_from_counts(self.counts)

