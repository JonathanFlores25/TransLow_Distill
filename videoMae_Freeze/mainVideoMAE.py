"""
mainVideoMAE.py
===============
Entry-point de entrenamiento para VideoMAE VAD Classifier.
Estructura basada en Step2_Detector/mainC2FPL.py.

Evaluación
----------
- Val  (cada época): test_Over()        — features .npy pre-extraídas (rápido)
- Test (cada época): test_Over_online() — video crudo → VideoMAE → clasificador
  sin guardar features a disco. Requiere --test_video_root.
"""

import os
import sys
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import VideoMAEImageProcessor, VideoMAEModel

from dataset   import Dataset_VideoMAE, BalancedBatchSampler
from model     import VideoMAE_VAD_Classifier
from train     import concatenated_train_feedback
from test      import test_Over, test_Over_online
from gpu_utils import get_optimal_device, try_init_rmm_pool, clear_gpu_cache
import option


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_in_one(epochs, train_loss, auc_val, ap_val,
                    auc_test=None, ap_test=None, save_path="curva_total.png"):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("AUC / AP")
    ax1.set_ylim(0, 1)
    ax1.plot(epochs, auc_val, label="auc_val")
    ax1.plot(epochs, ap_val,  label="ap_val", linestyle="--")
    if auc_test is not None:
        ax1.plot(epochs, auc_test, label="auc_test")
    if ap_test is not None:
        ax1.plot(epochs, ap_test,  label="ap_test", linestyle="--")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss")
    ax2.plot(epochs, train_loss, label="train_loss", linestyle=":")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.grid(True)
    plt.title("VideoMAE VAD — Loss + AUC/AP")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_curves(epochs, losses, auc_val, auc_test,
                         ap_val, ap_test, lrs, save_path):
    plt.figure()
    plt.plot(epochs, losses)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Train Loss")
    plt.grid(True)
    plt.savefig(save_path.replace(".png", "_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, auc_val,  label="AUC Val")
    plt.plot(epochs, auc_test, label="AUC Test")
    plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.title("AUC")
    plt.legend(); plt.grid(True)
    plt.savefig(save_path.replace(".png", "_auc.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, ap_val,  label="AP Val")
    plt.plot(epochs, ap_test, label="AP Test")
    plt.xlabel("Epoch"); plt.ylabel("AP"); plt.title("Average Precision")
    plt.legend(); plt.grid(True)
    plt.savefig(save_path.replace(".png", "_ap.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, lrs)
    plt.xlabel("Epoch"); plt.ylabel("LR"); plt.title("Learning Rate")
    plt.grid(True)
    plt.savefig(save_path.replace(".png", "_lr.png"))
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = option.parse_args()

    # ── GPU ──────────────────────────────────────────────────────────────────
    try_init_rmm_pool()
    device = get_optimal_device()

    # Multi-GPU: si hay más de una GPU visible (CUDA_VISIBLE_DEVICES=2,3),
    # se usa DataParallel para repartir batches entre ellas.
    gpu_ids      = list(range(torch.cuda.device_count()))
    use_multi_gpu = len(gpu_ids) > 1
    print(f"[main] Dispositivo principal: {device}  |  GPUs visibles: {gpu_ids}")

    # ── Backbone VideoMAE (congelado, para test_Over_online) ─────────────────
    use_online_test = (
        getattr(args, "test_video_root", None) is not None
        and getattr(args, "gt_annotation", None) is not None
    )

    backbone   = None
    processor  = None
    if use_online_test:
        print("[main] Cargando backbone VideoMAE para evaluación online del test set...")
        MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
        processor  = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
        backbone   = VideoMAEModel.from_pretrained(MODEL_NAME).to(device)
        if device.type == "cuda":
            backbone = backbone.half()   # fp16: mitad de VRAM
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        if use_multi_gpu:
            backbone = torch.nn.DataParallel(backbone, device_ids=gpu_ids)
            print(f"[main] Backbone en DataParallel GPUs {gpu_ids}.")
        backbone = torch.compile(backbone)
        print("[main] Backbone listo (frozen, fp16, compilado).")
    else:
        print("[main] --test_video_root no especificado. Test online desactivado.")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    _pin  = device.type == "cuda"
    _pf   = 4   # prefetch_factor: 4 batches adelantados por worker

    print("Cargando Val set...")
    val_loader = DataLoader(
        Dataset_VideoMAE(args, test_Mode="Validacion"),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=_pin,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=_pf if args.num_workers > 0 else None,
        drop_last=False,
    )

    print("Cargando Train set...")
    train_dataset = Dataset_VideoMAE(args, test_Mode="Train")
    labels        = np.asarray(train_dataset.labels_all).astype(int)
    train_loader  = DataLoader(
        train_dataset,
        batch_sampler=BalancedBatchSampler(labels, batch_size=args.batch_size, pos_fraction=0.5),
        num_workers=args.num_workers, pin_memory=_pin,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=_pf if args.num_workers > 0 else None,
    )
    print("Datasets listos.")

    # ── Clasificador ─────────────────────────────────────────────────────────
    model = VideoMAE_VAD_Classifier(feature_dim=768)
    model = model.to(device)
    if use_multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print(f"[main] Clasificador en DataParallel GPUs {gpu_ids}.")
    model = torch.compile(model)

    _model_core = model.module if use_multi_gpu else model
    total_params = sum(p.numel() for p in _model_core.parameters() if p.requires_grad)
    print(f"[main] Parámetros entrenables del clasificador: {total_params:,}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Optimizer, scheduler y AMP scaler ───────────────────────────────────
    optimizer = optim.SGD(_model_core.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Evaluación inicial (epoch 0) ─────────────────────────────────────────
    auc_val0, ap_val0 = test_Over(val_loader, model, args, device)
    if use_online_test:
        auc_test0, ap_test0 = test_Over_online(backbone, processor, model, args, device)
    else:
        auc_test0, ap_test0 = 0.0, 0.0
    print(f"Epoch 0 → auc_val={auc_val0:.4f}  auc_test={auc_test0:.4f}")

    # ── Historial ─────────────────────────────────────────────────────────────
    hist_epoch    = []
    hist_loss     = []
    hist_auc_val  = []
    hist_ap_val   = []
    hist_auc_test = []
    hist_ap_test  = []
    hist_lr       = []

    best_auc            = -1.0
    best_model_weights  = None

    # ── Loop de entrenamiento ─────────────────────────────────────────────────
    for epoch in tqdm(range(1, args.max_epoch + 1), total=args.max_epoch, dynamic_ncols=True):

        loss = concatenated_train_feedback(train_loader, model, optimizer, device, scaler)

        auc_val, ap_val = test_Over(val_loader, model, args, device)

        if use_online_test:
            auc_test, ap_test = test_Over_online(backbone, processor, model, args, device)
        else:
            auc_test, ap_test = auc_val, ap_val   # fallback: usar val como proxy

        scheduler.step(auc_val)
        lr = optimizer.param_groups[0]['lr']

        hist_epoch.append(epoch)
        hist_loss.append(loss)
        hist_auc_val.append(auc_val)
        hist_ap_val.append(ap_val)
        hist_auc_test.append(auc_test)
        hist_ap_test.append(ap_test)
        hist_lr.append(lr)

        print(
            f"\nEpoch {epoch}/{args.max_epoch} | "
            f"LR: {lr:.6f} | "
            f"aucVal: {auc_val:.4f} | aucTest: {auc_test:.4f} | "
            f"apVal: {ap_val:.4f} | apTest: {ap_test:.4f} | "
            f"loss: {loss:.4f}\n"
        )

        if auc_test > best_auc:
            best_auc           = auc_test
            best_model_weights = _model_core.state_dict().copy()
            ckpt_path = os.path.join(
                args.ckpt_dir,
                f"videomae_vad_{auc_test:.4f}_ep{epoch}.pt",
            )
            torch.save(best_model_weights, ckpt_path)

        plot_path = os.path.join(args.ckpt_dir, "training_curves.png")
        plot_training_curves(
            hist_epoch, hist_loss,
            hist_auc_val, hist_auc_test,
            hist_ap_val, hist_ap_test,
            hist_lr, plot_path,
        )
        plot_all_in_one(
            hist_epoch, hist_loss,
            hist_auc_val, hist_ap_val,
            hist_auc_test, hist_ap_test,
            save_path=os.path.join(args.ckpt_dir, "all_metrics.png"),
        )

    print(f"\n[main] Entrenamiento completo. Mejor AUC test: {best_auc:.4f}")
