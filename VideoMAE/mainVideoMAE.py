"""
mainC2FPL.py
============
Training entry-point for VideoMAE VAD Classifier (Step3).

Design rationale vs Step2_Detector:

- ALL logic inside if __name__ == '__main__': — no module-level side effects.
  Step2/mainC2FPL.py had the training loop OUTSIDE the if-guard (the `for
  epoch` loop was at module level, lines 202-250), meaning importing the
  module would trigger training. Fixed here by wrapping everything.

- Normalization stats computed on training split, saved to stats_dir, and
  passed to all dataset instances. This is new in Step3 because VideoMAE
  features have a domain shift from Kinetics→UCF-Crime that z-scoring
  per-feature partially corrects.

- Scheduler stepped every epoch with val_auc (ReduceLROnPlateau mode='max').
  Step2 called scheduler.step() sporadically. Here it is called exactly once
  per epoch, after validation, so LR decay is driven by the validation AUC
  plateau — the correct signal for anomaly detection tasks.

- Best checkpoint saved by val_auc (not test_auc). Step2 saved by test_auc
  which leaks test performance information into model selection. Fixing this
  to use val_auc is the standard practice for unbiased evaluation.

- Checkpoint format stores model_state + optimizer_state + epoch + val_auc
  so training can be resumed and checkpoints are self-documenting.

- Per-epoch table printed to stdout: epoch | loss | val_auc | test_auc | lr
  matches the format expected by operators monitoring long training runs.

- Plot saved as single .png with twin-axis (loss on right, AUC on left) plus
  separate per-metric plots — gives a quick visual summary of training health.

- pos_weight auto-computed from training label counts when args.pos_weight==-1.
  This correctly handles datasets where the anomaly/normal ratio differs from
  the 50/50 BalancedBatchSampler balance (the sampler balances per-batch but
  the true data distribution still informs the loss weighting).
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Relative imports from Step3_Classifier package ────────────────────────────
# All imports deferred to inside __main__ guard to prevent side effects.

if __name__ == "__main__":
    # ── Configure root logger before any other imports ────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    # ── Local module imports ───────────────────────────────────────────────────
    from dataset import (
        BalancedBatchSampler,
        VideoMAEFeatureDataset,
        collate_fn,
        compute_normalization_stats,
    )
    from gpu_utils import clear_gpu_cache, get_optimal_device, try_init_rmm_pool
    from model import VideoMAE_VAD_Classifier
    from option import parse_args
    from test import evaluate
    from train import train_one_epoch

    # ── Parse arguments ────────────────────────────────────────────────────────
    args = parse_args()

    # ── Reproducibility ────────────────────────────────────────────────────────
    # Seed Python, NumPy, and PyTorch for deterministic dataset shuffling and
    # weight initialisation. Note: DataLoader with num_workers>0 may still
    # produce non-deterministic ordering; setting worker_init_fn would fix that
    # but is omitted here for simplicity.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info("[main] Global seed set to %d.", args.seed)

    # ── Create output directories ──────────────────────────────────────────────
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.stats_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[main] Checkpoints: %s", args.ckpt_dir)
    logger.info("[main] Outputs    : %s", args.output_dir)
    logger.info("[main] Stats      : %s", args.stats_dir)

    # ── Device selection and optional RMM pool ─────────────────────────────────
    device = get_optimal_device(verbose=True)
    # RMM pool is beneficial on H100/A100 for large batches; silently skipped
    # if RAPIDS is not installed.
    if device.type == "cuda":
        try_init_rmm_pool(initial_gb=4.0)

    # ── Step 1: compute normalization statistics on TRAINING split only ────────
    # Load a raw (un-normalized) training dataset to compute Welford stats.
    # These stats will then be passed to ALL splits (train, val, test) so that
    # the same transformation is applied consistently.
    stats_mean_path = args.stats_dir / "feat_mean.npy"
    stats_std_path = args.stats_dir / "feat_std.npy"

    if stats_mean_path.exists() and stats_std_path.exists():
        logger.info(
            "[main] Loading existing normalization stats from %s ...", args.stats_dir
        )
        norm_stats = {
            "mean": np.load(stats_mean_path).astype(np.float32),
            "std": np.load(stats_std_path).astype(np.float32),
        }
    else:
        logger.info(
            "[main] Computing normalization stats on training split "
            "(Welford online algorithm) ..."
        )
        # stats=None here so we see raw Kinetics-domain features.
        raw_train_dataset = VideoMAEFeatureDataset(
            root_dir=args.features_dir,
            split="train",
            pseudo_label_dir=args.pseudo_label_dir,
            stats=None,
        )
        norm_stats = compute_normalization_stats(raw_train_dataset)
        np.save(stats_mean_path, norm_stats["mean"])
        np.save(stats_std_path, norm_stats["std"])
        logger.info(
            "[main] Stats saved: mean in [%.4f, %.4f], std in [%.4f, %.4f].",
            norm_stats["mean"].min(),
            norm_stats["mean"].max(),
            norm_stats["std"].min(),
            norm_stats["std"].max(),
        )
        # Free the raw dataset — we re-create it below with stats applied.
        del raw_train_dataset

    # ── Step 2: build datasets with normalization applied ─────────────────────
    logger.info("[main] Building datasets ...")

    train_dataset = VideoMAEFeatureDataset(
        root_dir=args.features_dir,
        split="train",
        pseudo_label_dir=args.pseudo_label_dir,
        stats=norm_stats,
    )
    val_dataset = VideoMAEFeatureDataset(
        root_dir=args.features_dir,
        split="val",
        stats=norm_stats,
    )
    test_dataset = VideoMAEFeatureDataset(
        root_dir=args.features_dir,
        split="test",
        stats=norm_stats,
    )

    # ── Step 3: compute pos_weight for BCELoss ─────────────────────────────────
    # pos_weight = n_neg / n_pos compensates for class imbalance beyond what
    # the BalancedBatchSampler already handles per batch.
    # The balanced sampler creates 50/50 mini-batches, but the true signal
    # magnitude in the loss should still reflect the dataset's anomaly rarity
    # so the model learns to be conservative about false positives.
    labels_np = train_dataset.label_array  # (N,) int32
    n_pos = int((labels_np == 1).sum())
    n_neg = int((labels_np == 0).sum())

    if args.pos_weight > 0:
        # Use user-specified value.
        pos_weight_val = args.pos_weight
        logger.info("[main] pos_weight (user-specified): %.4f", pos_weight_val)
    elif n_pos > 0:
        pos_weight_val = n_neg / n_pos
        logger.info(
            "[main] pos_weight (auto): n_neg=%d / n_pos=%d = %.4f",
            n_neg,
            n_pos,
            pos_weight_val,
        )
    else:
        pos_weight_val = 1.0
        logger.warning(
            "[main] Training split has 0 anomaly samples — pos_weight set to 1.0."
        )

    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion = nn.BCELoss(weight=None)
    # BCELoss does not have a native pos_weight parameter (that's BCEWithLogitsLoss).
    # We implement the equivalent manually: multiply the anomaly sample losses by
    # pos_weight inside the criterion wrapper below.
    # This keeps the model output as a direct probability (sigmoid applied in forward)
    # while still addressing class imbalance.
    # Using a wrapper class ensures the weighting is applied correctly per sample.

    class WeightedBCELoss(nn.Module):
        """
        BCELoss with per-class weighting equivalent to BCEWithLogitsLoss pos_weight.

        loss = -( pos_weight * y * log(p) + (1-y) * log(1-p) )

        WHY not BCEWithLogitsLoss: the model already applies Sigmoid in its
        forward() to produce probabilities, so logits are not directly available.
        This wrapper replicates pos_weight weighting on the probability output.
        """

        def __init__(self, pos_weight: torch.Tensor) -> None:
            super().__init__()
            self.pos_weight = pos_weight  # scalar tensor on device

        def forward(
            self, predictions: torch.Tensor, targets: torch.Tensor
        ) -> torch.Tensor:
            # Clamp to avoid log(0) even though Sigmoid output is (0,1) open interval.
            p = torch.clamp(predictions, min=1e-7, max=1.0 - 1e-7)
            loss = -(
                self.pos_weight * targets * torch.log(p)
                + (1.0 - targets) * torch.log(1.0 - p)
            )
            return loss.mean()

    criterion = WeightedBCELoss(pos_weight=pos_weight_tensor)

    # ── Step 4: build DataLoaders ──────────────────────────────────────────────
    pin_memory = device.type == "cuda"

    train_sampler = BalancedBatchSampler(
        labels=labels_np,
        batch_size=args.batch_size,
        pos_fraction=0.5,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 4,  # larger batch for faster eval
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )

    logger.info(
        "[main] Loaders — train batches: %d, val samples: %d, test samples: %d",
        len(train_sampler),
        len(val_dataset),
        len(test_dataset),
    )

    # ── Step 5: build model, optimizer, scheduler ──────────────────────────────
    # feature_dim assertion: validates that the args match the backbone.
    assert args.feature_dim == 768, (
        f"feature_dim must be 768 for VideoMAE ViT-Base. Got {args.feature_dim}."
    )

    model = VideoMAE_VAD_Classifier(feature_dim=args.feature_dim)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("[main] Model parameters (trainable): %d", total_params)

    # AdamW: decoupled weight decay is better than L2 regularisation for
    # models trained on transformer-derived feature spaces. The adaptive
    # moment estimation handles the varying gradient magnitudes across the
    # 768 entangled spatiotemporal dimensions more gracefully than SGD.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ReduceLROnPlateau with mode='max': the plateau metric is val_auc, and
    # the scheduler should REDUCE lr when the metric STOPS INCREASING.
    # patience=7 gives 7 epochs of stagnation before the first reduction —
    # appropriate for surveillance anomaly detection where val_auc can
    # fluctuate by ±0.01 from epoch to epoch.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=True,
    )

    # ── Step 6: training loop ──────────────────────────────────────────────────
    best_val_auc: float = -1.0
    best_ckpt_path: Path = args.ckpt_dir / "best_model.pt"

    # History lists for plotting.
    hist_epochs: list[int] = []
    hist_train_loss: list[float] = []
    hist_val_auc: list[float] = []
    hist_test_auc: list[float] = []
    hist_lr: list[float] = []

    logger.info("[main] Starting training for %d epochs ...", args.epochs)
    logger.info(
        "[main] {'epoch':>6} | {'loss':>8} | {'val_auc':>8} | {'test_auc':>9} | {'lr':>10}"
    )
    logger.info("[main] " + "-" * 55)

    for epoch in range(1, args.epochs + 1):

        # ── Training ──────────────────────────────────────────────────────────
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # ── Validation ────────────────────────────────────────────────────────
        val_auc, val_ap, val_fpr, val_tpr, val_prec, val_rec = evaluate(
            model, val_loader, args.gt_dir, args.clip_step, device
        )

        # ── Test ──────────────────────────────────────────────────────────────
        test_auc, test_ap, test_fpr, test_tpr, test_prec, test_rec = evaluate(
            model, test_loader, args.gt_dir, args.clip_step, device
        )

        # ── Scheduler step ─────────────────────────────────────────────────────
        # Stepped every epoch with val_auc — mandatory per design spec.
        # Step2 called scheduler.step() with test_auc and only sometimes,
        # which is both a data-leakage issue and a correctness bug.
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_auc)

        # ── Record history ────────────────────────────────────────────────────
        hist_epochs.append(epoch)
        hist_train_loss.append(train_loss)
        hist_val_auc.append(val_auc)
        hist_test_auc.append(test_auc)
        hist_lr.append(current_lr)

        # ── Per-epoch log line ────────────────────────────────────────────────
        logger.info(
            "[main] %6d | %8.4f | %8.4f | %9.4f | %10.2e",
            epoch,
            train_loss,
            val_auc,
            test_auc,
            current_lr,
        )

        # ── Best checkpoint by val_auc (NOT test_auc) ─────────────────────────
        # Using test_auc for model selection leaks test performance into the
        # training loop. The best model is the one that achieves the highest
        # validation AUC, evaluated independently at test time.
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "test_auc": test_auc,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            logger.info(
                "[main] >>> New best val_auc=%.4f at epoch %d — checkpoint saved.",
                val_auc,
                epoch,
            )

        # ── Save latest checkpoint (always) ───────────────────────────────────
        # Allows training to be resumed from the last epoch.
        latest_ckpt_path = args.ckpt_dir / "latest_model.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
                "test_auc": test_auc,
            },
            latest_ckpt_path,
        )

        # ── Save evaluation curves every 10 epochs ────────────────────────────
        if epoch % 10 == 0 or epoch == args.epochs:
            curves_dir = args.output_dir / f"epoch_{epoch:04d}"
            curves_dir.mkdir(parents=True, exist_ok=True)
            np.save(curves_dir / "val_fpr.npy", val_fpr)
            np.save(curves_dir / "val_tpr.npy", val_tpr)
            np.save(curves_dir / "val_precision.npy", val_prec)
            np.save(curves_dir / "val_recall.npy", val_rec)
            np.save(curves_dir / "test_fpr.npy", test_fpr)
            np.save(curves_dir / "test_tpr.npy", test_tpr)
            np.save(curves_dir / "test_precision.npy", test_prec)
            np.save(curves_dir / "test_recall.npy", test_rec)

        # ── Update training plot every epoch ─────────────────────────────────
        _save_training_plots(
            hist_epochs,
            hist_train_loss,
            hist_val_auc,
            hist_test_auc,
            hist_lr,
            save_dir=args.output_dir,
        )

        clear_gpu_cache()

    # ── Final summary ──────────────────────────────────────────────────────────
    logger.info("[main] " + "=" * 55)
    logger.info("[main] Training complete.")
    logger.info("[main] Best val_auc : %.4f", best_val_auc)
    logger.info("[main] Best checkpoint: %s", best_ckpt_path)

    # Save final test curves from best checkpoint.
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    final_test_auc, final_test_ap, fpr, tpr, prec, rec = evaluate(
        model, test_loader, args.gt_dir, args.clip_step, device
    )
    logger.info(
        "[main] Final test AUC (best checkpoint): %.4f  AP: %.4f",
        final_test_auc,
        final_test_ap,
    )
    final_curves_dir = args.output_dir / "final"
    final_curves_dir.mkdir(parents=True, exist_ok=True)
    np.save(final_curves_dir / "fpr.npy", fpr)
    np.save(final_curves_dir / "tpr.npy", tpr)
    np.save(final_curves_dir / "precision.npy", prec)
    np.save(final_curves_dir / "recall.npy", rec)
    logger.info("[main] Final curves saved to %s", final_curves_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helper — defined at module level so it can be called from __main__
# but does not execute at import time (it is a plain function, not a call).
# ─────────────────────────────────────────────────────────────────────────────

def _save_training_plots(
    epochs: list,
    train_loss: list,
    val_auc: list,
    test_auc: list,
    lr: list,
    save_dir: Path,
) -> None:
    """
    Save training progress plots to save_dir.

    Generates:
    - training_curves.png : combined twin-axis plot (AUC left, loss right)
    - training_loss.png   : train loss only
    - training_auc.png    : val_auc and test_auc
    - training_lr.png     : learning rate schedule

    All figures are saved in non-interactive backend mode so this function
    is safe to call from a subprocess or headless server.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless backend — must be set before pyplot import
    import matplotlib.pyplot as plt

    ep = list(epochs)

    # ── Combined twin-axis plot ────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("AUC", color="tab:blue")
    ax1.set_ylim(0.0, 1.0)
    ax1.plot(ep, val_auc, label="val_auc", color="tab:blue", linewidth=1.5)
    ax1.plot(
        ep, test_auc, label="test_auc", color="tab:cyan",
        linewidth=1.5, linestyle="--"
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Train Loss", color="tab:red")
    ax2.plot(ep, train_loss, label="train_loss", color="tab:red", linewidth=1.0, alpha=0.7)
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.title("VideoMAE VAD Classifier — Training Progress")
    fig.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=120)
    plt.close(fig)

    # ── Loss only ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, train_loss, color="tab:red", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCELoss")
    ax.set_title("Train Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_dir / "training_loss.png", dpi=120)
    plt.close(fig)

    # ── AUC only ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, val_auc, label="val_auc", linewidth=1.5)
    ax.plot(ep, test_auc, label="test_auc", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Frame-level AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_dir / "training_auc.png", dpi=120)
    plt.close(fig)

    # ── Learning rate ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, lr, color="tab:green", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.set_title("Learning Rate (ReduceLROnPlateau)")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    plt.savefig(save_dir / "training_lr.png", dpi=120)
    plt.close(fig)
