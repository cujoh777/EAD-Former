import os
import math
import argparse
import csv
import json
import sys
from datetime import datetime
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from core.datasets import LEVIRCD_Dataset
from core.models import (
    SwinEADFormer,
    FCEF,
    FCSiamConc,
    FCSiamDiff,
    BIT_Mini,
    ChangeFormerV6,
    SNUNetCD,
    FCCDN,
    Changer,
    ChangeMamba,
    EdgeCVT,
    CDMamba,
)
from core.utils import (
    set_seed,
    seed_worker,
    EADLoss,
    print_vram_usage
)


class TeeLogger:
    def __init__(self, log_path, quiet_console=False):
        self.console = sys.__stdout__
        self.file = open(log_path, "a", encoding="utf-8", buffering=1)
        self.quiet_console = quiet_console

    def write(self, text):
        if not self.quiet_console:
            self.console.write(text)
        self.file.write(text)

    def flush(self):
        if not self.quiet_console:
            self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def setup_text_logging(log_path, quiet_console=False):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = TeeLogger(log_path, quiet_console=quiet_console)
    sys.stdout = logger
    sys.stderr = logger
    return logger


def make_progress(loader, desc, disabled=False):
    return tqdm(
        loader,
        desc=desc,
        leave=False,
        ncols=110,
        dynamic_ncols=False,
        mininterval=1.0,
        ascii=True,
        disable=disabled
    )


def append_epoch_log(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_new_file = not os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path,
    epoch,
    best_f1,
    model,
    optimizer,
    scheduler,
    scaler,
    args
):
    checkpoint = {
        "epoch": epoch,
        "best_f1": best_f1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args)
    }
    torch.save(checkpoint, path)
    return checkpoint


# =====================================================
# Metrics
# =====================================================
class MetricsTracker:
    """
    二分类变化检测指标统计器。
    注意：这里直接放在 train.py 中，避免和 utils.py 版本不一致。
    """
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
        """
        preds: sigmoid 后的概率图，范围 [0, 1]
        labels: 二值标签，范围 {0, 1}
        """
        preds = (preds > self.threshold).int()
        labels = (labels > 0.5).int()

        self.TP += ((preds == 1) & (labels == 1)).sum().item()
        self.FP += ((preds == 1) & (labels == 0)).sum().item()
        self.TN += ((preds == 0) & (labels == 0)).sum().item()
        self.FN += ((preds == 0) & (labels == 1)).sum().item()

    def get_metrics(self):
        eps = 1e-8

        precision = self.TP / (self.TP + self.FP + eps)
        recall = self.TP / (self.TP + self.FN + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        iou = self.TP / (self.TP + self.FP + self.FN + eps)
        oa = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN + eps)

        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "IoU": iou,
            "OA": oa
        }


# =====================================================
# Mask preprocess
# =====================================================
def prepare_mask(mask):
    """
    将 LEVIR-CD 的 mask 统一整理为 [B, 1, H, W]，取值 {0, 1}。
    兼容：
    - [B, H, W]
    - [B, 1, H, W]
    - [B, H, W, 1]
    - 0/1 或 0/255
    """
    mask = mask.float()

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    if mask.dim() == 4 and mask.shape[-1] == 1:
        mask = mask.permute(0, 3, 1, 2).contiguous()

    mask = torch.nan_to_num(
        mask,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    if mask.max() > 1.0:
        mask = mask / 255.0

    mask = mask.clamp(0.0, 1.0)
    mask = (mask > 0.5).float()

    return mask


# =====================================================
# Edge GT
# =====================================================
_EDGE_KERNEL = None


def build_edge_gt(mask):
    """
    用拉普拉斯算子从二值变化 mask 中构建边缘监督。
    输入 mask: [B, 1, H, W]，取值 {0, 1}
    输出 edge: [B, 1, H, W]，取值 {0, 1}
    """
    global _EDGE_KERNEL

    if _EDGE_KERNEL is None or _EDGE_KERNEL.device != mask.device:
        _EDGE_KERNEL = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]],
            dtype=torch.float32,
            device=mask.device
        ).view(1, 1, 3, 3)

    kernel = _EDGE_KERNEL.to(dtype=mask.dtype)

    edge = F.conv2d(mask, kernel, padding=1).abs()
    edge = (edge > 0).float()

    return edge


# =====================================================
# Loss
# =====================================================
def compute_loss(outputs, labels, criterion, aux_weight, edge_weight, sparse_weight):
    """
    最终版 loss 计算。
    关键点：
    1. preds / aux 是 logits，所以 criterion 内部可以用 BCEWithLogits。
    2. edge 是模型返回的概率图，不是 logits，所以 edge loss 使用 BCE。
    3. BCE 对输入范围非常敏感，所以 edge 先 nan_to_num，再 clamp。
    4. loss 计算放在 autocast 外部，整体使用 float32，更稳定。
    """
    preds = outputs["pred"].float()
    aux = outputs["aux"].float()
    edge = outputs["edge"].float()
    sparse_loss = outputs["sparsity_loss"]

    labels = labels.float()
    labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
    labels = labels.clamp(0.0, 1.0)

    edge_gt = build_edge_gt(labels).float()
    edge_gt = torch.nan_to_num(edge_gt, nan=0.0, posinf=1.0, neginf=0.0)
    edge_gt = edge_gt.clamp(0.0, 1.0)

    loss_main = criterion(preds, labels)
    loss_aux = criterion(aux, labels)

    # edge 是概率图，不是 logits。
    # binary_cross_entropy 要求输入必须在 [0, 1]，且不能有 NaN / Inf。
    with torch.amp.autocast(device_type="cuda", enabled=False):
        edge_prob = edge.float()
        edge_prob = torch.nan_to_num(
            edge_prob,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )
        edge_prob = edge_prob.clamp(1e-6, 1.0 - 1e-6)

        loss_edge = F.binary_cross_entropy(
            edge_prob,
            edge_gt.float()
        )

    if torch.is_tensor(sparse_loss):
        sparse_loss = sparse_loss.float()
        sparse_loss = torch.nan_to_num(
            sparse_loss,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
    else:
        sparse_loss = torch.tensor(
            sparse_loss,
            device=labels.device,
            dtype=torch.float32
        )

    loss_sparse = sparse_weight * sparse_loss

    loss = (
        loss_main
        + aux_weight * loss_aux
        + edge_weight * loss_edge
        + loss_sparse
    )

    loss_items = {
        "loss_main": loss_main.detach(),
        "loss_aux": loss_aux.detach(),
        "loss_edge": loss_edge.detach(),
        "loss_sparse": loss_sparse.detach(),
        "loss_total": loss.detach(),
        "edge_min": edge_prob.detach().min(),
        "edge_max": edge_prob.detach().max()
    }

    return loss, loss_items


# =====================================================
# Train one epoch
# =====================================================
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    accum_steps,
    threshold,
    use_amp,
    aux_weight,
    edge_weight,
    sparse_weight,
    disable_tqdm=False
):
    model.train()

    metrics = MetricsTracker(threshold=threshold)
    total_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    pbar = make_progress(loader, "[Train]", disabled=disable_tqdm)

    for batch_idx, (t1, t2, labels) in enumerate(pbar):
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        labels = prepare_mask(labels.to(device, non_blocking=True))

        # 只让模型前向进入 AMP，loss 在外面用 float32 算
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(t1, t2)

        loss_raw, loss_items = compute_loss(
            outputs,
            labels,
            criterion,
            aux_weight=aux_weight,
            edge_weight=edge_weight,
            sparse_weight=sparse_weight
        )
        loss = loss_raw / accum_steps

        if not torch.isfinite(loss_raw):
            print("===== NAN / INF DETECTED =====")

            for k, v in loss_items.items():
                if torch.is_tensor(v):
                    print(k, v.detach().item())
                else:
                    print(k, v)

            pred = outputs["pred"].detach()
            edge = outputs["edge"].detach()

            print("pred max/min:", pred.max().item(), pred.min().item())
            print("edge max/min:", edge.max().item(), edge.min().item())
            print("edge nan count:", torch.isnan(edge).sum().item())
            print("edge inf count:", torch.isinf(edge).sum().item())

            raise RuntimeError("Loss became NaN or Inf")

        scaler.scale(loss).backward()

        do_step = (
            (batch_idx + 1) % accum_steps == 0
            or batch_idx + 1 == len(loader)
        )

        if do_step:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        total_loss += loss_raw.detach().item()

        with torch.no_grad():
            pred_prob = torch.sigmoid(outputs["pred"].detach().float())
            metrics.update(pred_prob, labels.detach().float())

        pbar.set_postfix(
            loss=f"{loss_raw.detach().item():.4f}",
            edge=f"{loss_items['edge_min'].item():.3f}-{loss_items['edge_max'].item():.3f}"
        )

    avg_loss = total_loss / len(loader)

    return avg_loss, metrics.get_metrics()


# =====================================================
# Validate
# =====================================================
@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    threshold,
    use_amp,
    aux_weight,
    edge_weight,
    sparse_weight,
    disable_tqdm=False
):
    model.eval()

    metrics = MetricsTracker(threshold=threshold)
    total_loss = 0.0

    pbar = make_progress(loader, "[Val]", disabled=disable_tqdm)

    for t1, t2, labels in pbar:
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        labels = prepare_mask(labels.to(device, non_blocking=True))

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(t1, t2)

        loss_raw, loss_items = compute_loss(
            outputs,
            labels,
            criterion,
            aux_weight=aux_weight,
            edge_weight=edge_weight,
            sparse_weight=sparse_weight
        )

        total_loss += loss_raw.detach().item()

        pred_prob = torch.sigmoid(outputs["pred"].detach().float())
        metrics.update(pred_prob, labels.detach().float())

        pbar.set_postfix(
            loss=f"{loss_raw.detach().item():.4f}"
        )

    avg_loss = total_loss / len(loader)

    return avg_loss, metrics.get_metrics()


# =====================================================
# Optimizer groups
# =====================================================
def get_parameter_groups(model, base_lr, weight_decay):
    """
    backbone 使用较小学习率，其余新模块使用 base_lr。
    bias / norm 不做权重衰减。
    """
    groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lr = base_lr * 0.1 if "backbone" in name else base_lr

        no_decay = any(
            key in name.lower()
            for key in ["bias", "norm", "bn", "relative_position_bias_table"]
        )

        wd = 0.0 if no_decay else weight_decay

        groups.append({
            "params": [param],
            "lr": lr,
            "weight_decay": wd
        })

    return groups


# =====================================================
# Build model
# =====================================================
def build_model(
    model_name,
    device,
    symmetric=True,
    router_mode="ead",
    router_cues="ers",
    fusion_mode="phi",
    gate_position="after",
):
    if model_name == "FCEF":
        model = FCEF()
    elif model_name == "FCSiamConc":
        model = FCSiamConc()
    elif model_name == "FCSiamDiff":
        model = FCSiamDiff()
    elif model_name == "BIT":
        model = BIT_Mini()
    elif model_name == "ChangeFormer":
        model = ChangeFormerV6()
    elif model_name in {"SNUNet_CD", "SNUNetCD"}:
        model = SNUNetCD()
    elif model_name == "FCCDN":
        model = FCCDN()
    elif model_name == "Changer":
        model = Changer()
    elif model_name == "ChangeMamba":
        model = ChangeMamba()
    elif model_name == "EdgeCVT":
        model = EdgeCVT()
    elif model_name == "CDMamba":
        model = CDMamba()
    elif model_name == "SwinEADFormer":
        model = SwinEADFormer(
            symmetric=symmetric,
            router_mode=router_mode,
            router_cues=router_cues,
            fusion_mode=fusion_mode,
            gate_position=gate_position,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model.to(device)


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="SwinEADFormer",
        choices=[
            "SwinEADFormer",
            "FCEF",
            "FCSiamConc",
            "FCSiamDiff",
            "BIT",
            "ChangeFormer",
            "SNUNet_CD",
            "SNUNetCD",
            "FCCDN",
            "Changer",
            "ChangeMamba",
            "EdgeCVT",
            "CDMamba",
        ]
    )

    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="Path to the prepared dataset root. No dataset path is bundled."
    )

    parser.add_argument(
        "--epochs",
        default=100,
        type=int
    )

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int
    )

    parser.add_argument(
        "--lr",
        default=3e-5,
        type=float
    )

    parser.add_argument(
        "--weight_decay",
        default=1e-2,
        type=float
    )

    parser.add_argument(
        "--accum_steps",
        default=1,
        type=int
    )

    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float
    )

    parser.add_argument(
        "--aux_weight",
        default=0.4,
        type=float,
        help="Auxiliary prediction loss weight."
    )

    parser.add_argument(
        "--edge_weight",
        default=0.2,
        type=float,
        help="Router edge supervision loss weight."
    )

    parser.add_argument(
        "--sparse_weight",
        default=1e-4,
        type=float,
        help="Router sparsity regularization weight."
    )

    parser.add_argument(
        "--no_symmetric",
        action="store_true",
        help="Disable bidirectional symmetric EAD fusion."
    )

    parser.add_argument(
        "--router_mode",
        default="ead",
        choices=["ead", "none"],
        help="Use EAD router or an all-one router mask."
    )

    parser.add_argument(
        "--router_cues",
        default="ers",
        choices=["ers", "rs", "es", "er", "e", "r", "s"],
        help="Router cue subset: e=edge, r=region, s=semantic."
    )
    parser.add_argument(
        "--fusion_mode",
        default="phi",
        choices=["phi", "interaction_only"],
        help="Use full Phi fusion or only bidirectional interaction difference."
    )
    parser.add_argument(
        "--gate_position",
        default="after",
        choices=["before", "after"],
        help="Apply the router before attention inputs or after attention outputs."
    )
    parser.add_argument(
        "--variant",
        default="full",
        help="Stable experiment label stored in checkpoints and metric files."
    )

    parser.add_argument(
        "--num_workers",
        default=8,
        type=int
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP mixed precision training. Default is disabled for stability."
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic cuDNN settings and seeded DataLoader workers."
    )

    parser.add_argument(
        "--save_dir",
        default="checkpoints",
        type=str
    )

    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Disable progress bars for nohup, log files, or narrow remote terminals."
    )

    parser.add_argument(
        "--log_csv",
        default=None,
        type=str,
        help="CSV path for epoch metrics. Default: <save_dir>/<model>_metrics.csv"
    )

    parser.add_argument(
        "--log_dir",
        default="logs",
        type=str,
        help="Directory for text and CSV training logs."
    )

    parser.add_argument(
        "--log_file",
        default=None,
        type=str,
        help="Text log path. Default: <log_dir>/<run_name>.log"
    )

    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        help="Experiment name used for log file names."
    )

    parser.add_argument(
        "--quiet_console",
        action="store_true",
        help="Write logs to files without printing them in the current terminal."
    )

    args = parser.parse_args()

    args.run_name = args.run_name or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(args.log_dir, f"{args.run_name}.log")
    log_csv = args.log_csv or os.path.join(args.log_dir, f"{args.run_name}_metrics.csv")
    text_logger = setup_text_logging(log_file, quiet_console=args.quiet_console)

    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(args.save_dir, exist_ok=True)

    runtime_config = vars(args).copy()
    runtime_config.update({
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    })
    config_path = os.path.join(args.save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(runtime_config, f, indent=2, sort_keys=True)

    print("=" * 60)
    print(f">>> Run name: {args.run_name}")
    print(f">>> Training model: {args.model}")
    print(f">>> Device: {device}")
    print(f">>> Data dir: {args.data_dir}")
    print(f">>> Epochs: {args.epochs}")
    print(f">>> Batch size: {args.batch_size}")
    print(f">>> LR: {args.lr}")
    print(f">>> Threshold: {args.threshold}")
    print(f">>> Aux weight: {args.aux_weight}")
    print(f">>> Edge weight: {args.edge_weight}")
    print(f">>> Sparse weight: {args.sparse_weight}")
    print(f">>> Symmetric: {not args.no_symmetric}")
    print(f">>> Router mode: {args.router_mode}")
    print(f">>> Router cues: {args.router_cues}")
    print(f">>> Fusion mode: {args.fusion_mode}")
    print(f">>> Gate position: {args.gate_position}")
    print(f">>> Variant: {args.variant}")
    print(f">>> Seed: {args.seed}")
    print(f">>> Deterministic: {args.deterministic}")
    print(f">>> Config: {config_path}")
    print(f">>> AMP: {args.amp}")
    print(f">>> Text log: {log_file}")
    print(f">>> Metrics CSV: {log_csv}")
    print("=" * 60)

    model = build_model(
        args.model,
        device,
        symmetric=not args.no_symmetric,
        router_mode=args.router_mode,
        router_cues=args.router_cues,
        fusion_mode=args.fusion_mode,
        gate_position=args.gate_position,
    )

    criterion = EADLoss(
        alpha=0.5,
        gamma=2.0,
        boundary_weight=0.4
    )

    use_amp = bool(args.amp and device.type == "cuda")

    scaler = GradScaler(
        "cuda",
        enabled=use_amp
    )

    optimizer = torch.optim.AdamW(
        get_parameter_groups(
            model,
            base_lr=args.lr,
            weight_decay=args.weight_decay
        )
    )

    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        progress = float(epoch - warmup_epochs) / float(
            max(1, args.epochs - warmup_epochs)
        )

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )

    ckpt_path = os.path.join(args.save_dir, f"{args.model}_latest.pth")
    best_path = os.path.join(args.save_dir, f"{args.model}_best.pth")

    start_epoch = 0
    best_f1 = 0.0

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(
            ckpt_path,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0.0)

        print(f">>> Resume from epoch {start_epoch}")
        print(f">>> Current best F1: {best_f1:.4f}")

    num_workers = min(args.num_workers, os.cpu_count() or args.num_workers)

    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    )

    if num_workers > 0:
        loader_args.update(
            persistent_workers=True,
            prefetch_factor=2
        )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed + 1)

    train_loader = DataLoader(
        LEVIRCD_Dataset(args.data_dir, "train"),
        shuffle=True,
        generator=train_generator,
        **loader_args
    )

    val_loader = DataLoader(
        LEVIRCD_Dataset(args.data_dir, "val"),
        shuffle=False,
        generator=val_generator,
        **loader_args
    )

    for epoch in range(start_epoch, args.epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            accum_steps=args.accum_steps,
            threshold=args.threshold,
            use_amp=use_amp,
            aux_weight=args.aux_weight,
            edge_weight=args.edge_weight,
            sparse_weight=args.sparse_weight,
            disable_tqdm=args.disable_tqdm
        )

        val_loss, val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
            use_amp=use_amp,
            aux_weight=args.aux_weight,
            edge_weight=args.edge_weight,
            sparse_weight=args.sparse_weight,
            disable_tqdm=args.disable_tqdm
        )

        scheduler.step()

        current_lr = optimizer.param_groups[-1]["lr"]

        print("\n" + "=" * 60)
        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(f"LR={current_lr:.8f}")

        print(
            f"Train Loss={train_loss:.4f} "
            f"IoU={train_metrics['IoU']:.4f} "
            f"F1={train_metrics['F1']:.4f}"
        )

        print(
            f"Val   Loss={val_loss:.4f} "
            f"IoU={val_metrics['IoU']:.4f} "
            f"F1={val_metrics['F1']:.4f} "
            f"P={val_metrics['Precision']:.4f} "
            f"R={val_metrics['Recall']:.4f} "
            f"OA={val_metrics['OA']:.4f}"
        )

        if device.type == "cuda":
            print_vram_usage()

        is_best = val_metrics["F1"] > best_f1

        append_epoch_log(
            log_csv,
            {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_iou": train_metrics["IoU"],
                "train_f1": train_metrics["F1"],
                "val_loss": val_loss,
                "val_iou": val_metrics["IoU"],
                "val_f1": val_metrics["F1"],
                "val_precision": val_metrics["Precision"],
                "val_recall": val_metrics["Recall"],
                "val_oa": val_metrics["OA"],
                "best_f1": max(best_f1, val_metrics["F1"]),
                "is_best": int(is_best)
            }
        )

        if is_best:
            best_f1 = val_metrics["F1"]

        checkpoint = save_checkpoint(
            ckpt_path,
            epoch,
            best_f1,
            model,
            optimizer,
            scheduler,
            scaler,
            args
        )

        if is_best:
            torch.save(checkpoint, best_path)
            print(f"🌟 New Best F1: {best_f1:.4f}")

        print("=" * 60)


if __name__ == "__main__":
    main()
