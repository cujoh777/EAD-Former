import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core import models as model_zoo
from core.models import SwinEADFormer, FCEF, FCSiamConc, FCSiamDiff, BIT_Mini, ChangeFormerV6
from core.boundary_metrics import BoundaryMetricTracker
from core.datasets import LEVIRCD_Dataset
from core.utils import MetricsTracker, prepare_mask, seed_worker


def build_model(
    name,
    symmetric=True,
    router_mode="ead",
    router_cues="ers",
    fusion_mode="phi",
    gate_position="after",
):
    if name == "FCEF":
        return FCEF()
    if name == "FCSiamConc":
        return FCSiamConc()
    if name == "FCSiamDiff":
        return FCSiamDiff()
    if name == "BIT":
        return BIT_Mini()
    if name == "ChangeFormer":
        return ChangeFormerV6()
    if name in {"SNUNet_CD", "SNUNetCD"}:
        cls = getattr(model_zoo, "SNUNetCD", None)
        if cls is None:
            raise ValueError("SNUNetCD is not registered in core.models. Upload the updated core/models.py first.")
        return cls()
    if name == "FCCDN":
        cls = getattr(model_zoo, "FCCDN", None)
        if cls is None:
            raise ValueError("FCCDN is not registered in core.models. Upload the updated core/models.py first.")
        return cls()
    if name == "Changer":
        cls = getattr(model_zoo, "Changer", None)
        if cls is None:
            raise ValueError("Changer is not registered in core.models. Upload the updated core/models.py first.")
        return cls()
    if name == "ChangeMamba":
        cls = getattr(model_zoo, "ChangeMamba", None)
        if cls is None:
            raise ValueError("ChangeMamba is not registered in core.models. Upload the updated core/models.py first.")
        return cls()
    if name == "EdgeCVT":
        cls = getattr(model_zoo, "EdgeCVT", None)
        if cls is None:
            raise ValueError("EdgeCVT is not registered in core.models. Upload the updated core/models.py first.")
        return cls()
    if name == "CDMamba":
        cls = getattr(model_zoo, "CDMamba", None)
        if cls is None:
            raise ValueError("CDMamba is not registered in core.models. Upload the updated core/models.py first.")
        return cls()
    return SwinEADFormer(
        symmetric=symmetric,
        router_mode=router_mode,
        router_cues=router_cues,
        fusion_mode=fusion_mode,
        gate_position=gate_position,
    )


def load_checkpoint(weights_path, device):
    return torch.load(weights_path, map_location=device, weights_only=False)


def load_model_weights(model, ckpt):
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    return ckpt if isinstance(ckpt, dict) else {}


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    threshold,
    boundary_tolerance,
    boundary_kernel_size,
    boundary_mode,
):
    model.eval()
    meter = MetricsTracker(threshold=threshold)
    boundary_meter = BoundaryMetricTracker(
        tolerance=boundary_tolerance,
        boundary_kernel_size=boundary_kernel_size,
        boundary_mode=boundary_mode,
    )

    for t1, t2, labels in tqdm(loader, desc="Evaluate"):
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        labels = prepare_mask(labels.to(device, non_blocking=True))

        outputs = model(t1, t2)
        preds = torch.sigmoid(outputs["pred"])
        meter.update(preds, labels)
        pred_bin = (preds > threshold).float()
        target_bin = (labels > 0.5).float()
        boundary_meter.update(pred_bin.detach().cpu(), target_bin.detach().cpu())

    metrics = meter.get_metrics()
    metrics.update(boundary_meter.compute())
    return metrics


def save_metrics_csv(path, metrics):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


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
        "--data-dir",
        required=True,
        help="Path to the prepared dataset root. No dataset path is bundled."
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--weights", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary_tolerance", type=int, default=2)
    parser.add_argument("--boundary_kernel_size", type=int, default=3)
    parser.add_argument(
        "--boundary_mode",
        choices=["inner", "symmetric"],
        default="inner",
    )
    parser.add_argument("--save_metrics_csv", default="")
    parser.add_argument("--no_symmetric", action="store_true")
    parser.add_argument("--router_mode", default=None, choices=["ead", "none"])
    parser.add_argument(
        "--router_cues",
        default=None,
        choices=["ers", "rs", "es", "er", "e", "r", "s"]
    )
    parser.add_argument("--fusion_mode", default=None, choices=["phi", "interaction_only"])
    parser.add_argument("--gate_position", default=None, choices=["before", "after"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = args.weights or os.path.join("checkpoints", f"{args.model}_best.pth")
    ckpt = load_checkpoint(weights_path, device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    symmetric = not args.no_symmetric
    if "no_symmetric" in ckpt_args and not args.no_symmetric:
        symmetric = not ckpt_args["no_symmetric"]

    router_mode = args.router_mode or ckpt_args.get("router_mode", "ead")
    router_cues = args.router_cues or ckpt_args.get("router_cues", "ers")
    fusion_mode = args.fusion_mode or ckpt_args.get("fusion_mode", "phi")
    gate_position = args.gate_position or ckpt_args.get("gate_position", "after")

    print(
        f"\nStart eval: model={args.model}, split={args.split}, "
        f"threshold={args.threshold}, symmetric={symmetric}, "
        f"router={router_mode}, router_cues={router_cues}, "
        f"fusion={fusion_mode}, gate_position={gate_position}"
    )
    model = build_model(
        args.model,
        symmetric=symmetric,
        router_mode=router_mode,
        router_cues=router_cues,
        fusion_mode=fusion_mode,
        gate_position=gate_position,
    ).to(device)
    ckpt = load_model_weights(model, ckpt)
    if "best_f1" in ckpt:
        print(f"Loaded checkpoint best_f1={ckpt['best_f1']:.4f}")

    num_workers = min(8, os.cpu_count() or 0)
    eval_generator = torch.Generator()
    eval_generator.manual_seed(int(ckpt_args.get("seed", 42)) + 2)
    loader = DataLoader(
        LEVIRCD_Dataset(args.data_dir, mode=args.split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=eval_generator,
    )

    metrics = evaluate(
        model,
        loader,
        device,
        args.threshold,
        args.boundary_tolerance,
        args.boundary_kernel_size,
        args.boundary_mode,
    )

    print("\n" + "=" * 50)
    print(f"{args.model} {args.split} metrics")
    print("=" * 50)
    print(f"Precision: {metrics['Precision'] * 100:.2f}%")
    print(f"Recall   : {metrics['Recall'] * 100:.2f}%")
    print(f"F1-Score : {metrics['F1'] * 100:.2f}%")
    print(f"IoU      : {metrics['IoU'] * 100:.2f}%")
    print(f"OA       : {metrics['OA'] * 100:.2f}%")
    print(f"B-Prec.  : {metrics['B-Precision'] * 100:.2f}%")
    print(f"B-Recall : {metrics['B-Recall'] * 100:.2f}%")
    print(f"B-F1     : {metrics['B-F1'] * 100:.2f}%")
    print(f"B-IoU    : {metrics['B-IoU'] * 100:.2f}%")
    print("=" * 50)

    if args.save_metrics_csv:
        csv_metrics = {
            "model": args.model,
            "split": args.split,
            "threshold": args.threshold,
            "weights": weights_path,
            "symmetric": symmetric,
            "router_mode": router_mode,
            "router_cues": router_cues,
            "fusion_mode": fusion_mode,
            "gate_position": gate_position,
            "variant": ckpt_args.get("variant", ""),
            "seed": ckpt_args.get("seed", ""),
            "run_name": ckpt_args.get("run_name", ""),
            "deterministic": ckpt_args.get("deterministic", ""),
            "boundary_tolerance": args.boundary_tolerance,
            "boundary_kernel_size": args.boundary_kernel_size,
            "boundary_mode": args.boundary_mode,
        }
        csv_metrics.update({key: value * 100.0 for key, value in metrics.items()})
        save_metrics_csv(args.save_metrics_csv, csv_metrics)
        print(f"Saved metrics csv: {args.save_metrics_csv}")


if __name__ == "__main__":
    main()
