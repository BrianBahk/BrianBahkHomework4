# This code was written by GPT-5
"""
Train planners (MLP, Transformer, ViT) on the SuperTuxKart drive dataset.

Example:
    python -m homework.train_planner --model mlp_planner --epochs 20 --batch_size 256
    python -m homework.train_planner --model vit_planner --epochs 30 --batch_size 64 --scheduler cosine --warmup_epochs 3 --amp
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import MODEL_FACTORY, save_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    l1 = (pred - target).abs()
    l1 = l1 * mask[..., None].float()
    denom = mask.sum().clamp(min=1).float()
    return l1.sum() / denom


def run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    optimizer=None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    metric = PlannerMetric()
    model.train(train_mode)

    for batch in tqdm(loader, leave=False):
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        if "image" in batch:
            preds = model(batch["image"])  # ViT
        else:
            preds = model(batch["track_left"], batch["track_right"])  # MLP/Transformer

        labels = batch["waypoints"]
        mask = batch["waypoints_mask"]

        if train_mode:
            if scaler is None:
                loss = masked_l1_loss(preds, labels, mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                with torch.cuda.amp.autocast():
                    loss = masked_l1_loss(preds, labels, mask)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

        metric.add(preds, labels, mask)

    return metric.compute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_FACTORY.keys()), default="mlp_planner")
    parser.add_argument("--train_dir", default=str(Path("drive_data/train")))
    parser.add_argument("--val_dir", default=str(Path("drive_data/val")))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine"], default="none")
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (CUDA only)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint if found")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--selection_metric",
        type=str,
        choices=["l1", "longitudinal", "lateral"],
        default="l1",
        help="Metric to select best checkpoint",
    )
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    transform_pipeline = "state_only" if args.model in ("mlp_planner", "transformer_planner") else "default"

    train_loader = load_data(
        args.train_dir,
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = load_data(
        args.val_dir,
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = MODEL_FACTORY[args.model]()
    model.to(device)

    if args.model == "vit_planner":
        args.lr = 3e-4 if args.lr == 2e-3 else args.lr
        args.batch_size = min(args.batch_size, 64)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Optional: resume from existing checkpoint
    ckpt_path = Path(__file__).resolve().parent / f"{args.model}.th"
    if args.resume and ckpt_path.exists():
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"Loaded checkpoint from {ckpt_path}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint: {e}")

    # Scheduler: warmup + cosine
    if args.scheduler == "cosine":
        def lr_lambda(epoch: int):
            # epoch is 0-indexed inside scheduler.step()
            e = epoch + 1
            if args.warmup_epochs > 0 and e <= args.warmup_epochs:
                return e / max(1, args.warmup_epochs)
            # cosine over remaining epochs
            total = max(1, args.epochs - args.warmup_epochs)
            prog = min(1.0, max(0.0, (e - args.warmup_epochs) / total))
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    # Track best value for chosen selection metric
    best_val = float("inf")
    best_path = None
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        if scheduler is not None:
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        train_metrics = run_epoch(model, train_loader, device, optimizer, scaler)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        print(
            f"  Train L1: {train_metrics['l1_error']:.4f} | Long: {train_metrics['longitudinal_error']:.4f} | Lat: {train_metrics['lateral_error']:.4f}"
        )
        print(
            f"  Val   L1: {val_metrics['l1_error']:.4f} | Long: {val_metrics['longitudinal_error']:.4f} | Lat: {val_metrics['lateral_error']:.4f}"
        )

        # Select metric
        sel_key = (
            "l1_error" if args.selection_metric == "l1" else (
                "longitudinal_error" if args.selection_metric == "longitudinal" else "lateral_error"
            )
        )
        current_val = val_metrics[sel_key]
        if current_val < best_val:
            best_val = current_val
            best_path = save_model(model)
            print(
                f"  Saved best model to {best_path} (best {sel_key}={best_val:.4f}; val L1={val_metrics['l1_error']:.4f})"
            )
            no_improve = 0
        else:
            no_improve += 1

        # Step scheduler at end of epoch
        if scheduler is not None:
            scheduler.step()

        # Early stopping
        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping after {epoch} epochs (no improvement for {no_improve} epochs)")
            break

    if best_path is None:
        best_path = save_model(model)
        print(f"Saved model to {best_path}")


if __name__ == "__main__":
    main()

