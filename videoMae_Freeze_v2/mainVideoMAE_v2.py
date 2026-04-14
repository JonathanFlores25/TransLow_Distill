"""
mainVideoMAE_v2.py
==================
Entry-point de entrenamiento — VideoMAE VAD Classifier v2 (Metric Learning).

Diferencias respecto a v1 (mainVideoMAE.py)
--------------------------------------------
  · Modelo       : VideoMAE_VAD_Classifier_v2 (encoder compartido + proj head)
  · Loss         : BCE + Label Smoothing + Soft Triplet con OHNM
  · Scheduler    : LinearLR (warmup) → CosineAnnealingLR
  · Evaluación   : usa model.predict() en lugar de model() para compatibilidad
  · Historial    : registra loss_bce y loss_triplet por separado
  · test.py      : importado de v1 sin modificaciones (usa model.predict)

Reutiliza de v1 (sin copiar)
-----------------------------
  videoMae_Freeze.dataset   — Dataset_VideoMAE, BalancedBatchSampler
  videoMae_Freeze.test      — test_Over, test_Over_online
  videoMae_Freeze.gpu_utils — get_optimal_device, try_init_rmm_pool
  uv run videoMae_Freeze_v2/mainVideoMAE_v2.py --features_dir data/vmae_features --clip_names_path data/pseudo_labels/clip_names.npy --gt_binary_path data/pseudo_labels/gt_binary.npy --gt_annotation resources/Temporal_Anomaly_Annotation.txt --test_video_root /media/pc/backup1/BaseDeDatos/UCF-Crime/Videos --ckpt_dir data/eval_results/vmae_v2_ckpt --output_dir data/eval_results/vmae_v2_output --stats_dir data/eval_results/vmae_v2_stats --max_epoch 100 --batch_size 64 --triplet_weight 0.5 --tau 0.5
"""

import copy
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

# ── Path setup ────────────────────────────────────────────────────────────────
# _ROOT  = raíz del repo  → permite importar videoMae_Freeze.* (v1)
# _HERE  = videoMae_Freeze_v2/  → permite importar model_v2, train_v2, etc.
_ROOT = Path(__file__).parent.parent
_HERE = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from videoMae_Freeze.dataset   import Dataset_VideoMAE
from videoMae_Freeze.test      import test_Over, test_Over_online
from videoMae_Freeze.gpu_utils import get_optimal_device, try_init_rmm_pool, clear_gpu_cache

from model_v2   import VideoMAE_VAD_Classifier_v2
from train_v2   import train_v2
from sampler_v2 import GuaranteedBalancedSampler
from viz_v2     import save_epoch_viz
import option_v2 as option


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch: test_Over y test_Over_online esperan model(x), no model.predict(x)
# Wrapeamos el modelo para que su __call__ devuelva solo el score.
# ─────────────────────────────────────────────────────────────────────────────

class _PredictWrapper(torch.nn.Module):
    """Envuelve el modelo v2 para que __call__ devuelva solo el score."""
    def __init__(self, model):
        super().__init__()
        self._m = model

    def forward(self, x):
        return self._m.predict(x)

    # Redirige train/eval al modelo real
    def train(self, mode=True):
        self._m.train(mode)
        return self

    def eval(self):
        self._m.eval()
        return self

    def state_dict(self, **kw):
        return self._m.state_dict(**kw)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(epochs, losses_total, losses_bce, losses_triplet,
                         auc_val, auc_test, ap_val, ap_test, lrs, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("VideoMAE VAD v2 — Metric Learning", fontsize=12)

    # Loss total + componentes
    ax = axes[0, 0]
    ax.plot(epochs, losses_total,   label="total",   lw=2)
    ax.plot(epochs, losses_bce,     label="bce",     ls="--")
    ax.plot(epochs, losses_triplet, label="triplet", ls=":")
    ax.set_title("Train Loss"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)

    # AUC
    ax = axes[0, 1]
    ax.plot(epochs, auc_val,  label="AUC Val")
    ax.plot(epochs, auc_test, label="AUC Test")
    ax.set_title("AUC"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)

    # AP
    ax = axes[1, 0]
    ax.plot(epochs, ap_val,  label="AP Val")
    ax.plot(epochs, ap_test, label="AP Test")
    ax.set_title("Average Precision"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)

    # LR
    ax = axes[1, 1]
    ax.plot(epochs, lrs)
    ax.set_title("Learning Rate"); ax.set_xlabel("Epoch"); ax.set_yscale("log")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    args = option.parse_args()

    # ── GPU ──────────────────────────────────────────────────────────────────
    try_init_rmm_pool()
    device = get_optimal_device()

    gpu_ids       = list(range(torch.cuda.device_count()))
    use_multi_gpu = len(gpu_ids) > 1
    print(f"[main] Dispositivo principal: {device}  |  GPUs visibles: {gpu_ids}")

    # ── Backbone VideoMAE (congelado, para test_Over_online) ─────────────────
    use_online_test = (
        getattr(args, "test_video_root", None) is not None
        and getattr(args, "gt_annotation", None) is not None
    )

    backbone  = None
    processor = None
    if use_online_test:
        print("[main] Cargando backbone VideoMAE para evaluación online...")
        MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
        processor  = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
        backbone   = VideoMAEModel.from_pretrained(MODEL_NAME).to(device)
        if device.type == "cuda":
            backbone = backbone.half()
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
    _pin = device.type == "cuda"
    _pf  = 4

    print("Cargando Val set...")
    val_loader = DataLoader(
        Dataset_VideoMAE(args, test_Mode="Validacion"),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=_pin,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=_pf if args.num_workers > 0 else None,
        drop_last=False,
    )

    print("Cargando Train set...")
    train_dataset = Dataset_VideoMAE(args, test_Mode="Train")
    labels_arr    = np.asarray(train_dataset.labels_all).astype(int)
    train_loader  = DataLoader(
        train_dataset,
        batch_sampler=GuaranteedBalancedSampler(
            labels_arr, batch_size=args.batch_size, pos_fraction=0.5,
            min_pos=2, min_neg=1,
        ),
        num_workers=args.num_workers,
        pin_memory=_pin,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=_pf if args.num_workers > 0 else None,
    )
    print("Datasets listos.")

    # ── Modelo v2 ─────────────────────────────────────────────────────────────
    model = VideoMAE_VAD_Classifier_v2(
        feature_dim=args.feature_dim,
        proj_hidden=args.proj_hidden,
        proj_out=args.proj_out,
    ).to(device)

    # Guardar referencia al modelo sin wrapper ANTES de DataParallel/compile,
    # para que _PredictWrapper pueda llamar .predict() y para state_dict/optimizer.
    _model_core = model

    if use_multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print(f"[main] Clasificador en DataParallel GPUs {gpu_ids}.")
    model = torch.compile(model)

    total_params = sum(p.numel() for p in _model_core.parameters() if p.requires_grad)
    print(f"[main] Parámetros entrenables: {total_params:,}")
    print(f"[main] Config: λ_t={args.triplet_weight}  τ={args.tau}  "
          f"smooth={args.label_smoothing}  warmup={args.warmup_epochs}")

    os.makedirs(args.ckpt_dir,   exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.stats_dir,  exist_ok=True)

    # Wrapper para compatibilidad con test_Over / test_Over_online de v1.
    # Usa _model_core (sin DataParallel) para acceder a .predict() directamente.
    eval_model = _PredictWrapper(_model_core)

    # ── Optimizer y AMP scaler ────────────────────────────────────────────────
    optimizer = optim.SGD(
        _model_core.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=0.9,
        nesterov=True,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Scheduler: warmup lineal → CosineAnnealing ───────────────────────────
    warmup_epochs = max(0, args.warmup_epochs)
    cosine_epochs = max(1, args.max_epoch - warmup_epochs)

    if warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_epochs,
                    eta_min=args.eta_min,
                ),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epoch, eta_min=args.eta_min
        )

    # ── args para test_Over en val set (sin gt_annotation) ──────────────────
    # El val set contiene videos de TRAIN que NO existen en
    # Temporal_Anomaly_Annotation.txt (que es el GT de test).
    # Sin gt_annotation, test_Over usa clip-level metrics con pseudo-labels
    # del directorio → AUC real en vez de 0.5000 trivial.
    val_args = copy.copy(args)
    val_args.gt_annotation = None

    # ── Evaluación inicial (epoch 0) ─────────────────────────────────────────
    auc_val0, ap_val0 = test_Over(val_loader, eval_model, val_args, device)
    if use_online_test:
        auc_test0, ap_test0 = test_Over_online(backbone, processor, eval_model, args, device)
    else:
        auc_test0, ap_test0 = 0.0, 0.0
    print(f"Epoch 0 → auc_val={auc_val0:.4f}  auc_test={auc_test0:.4f}")

    # ── Historial ─────────────────────────────────────────────────────────────
    hist = dict(
        epoch=[], loss_total=[], loss_bce=[], loss_triplet=[],
        auc_val=[], ap_val=[], auc_test=[], ap_test=[], lr=[],
    )
    best_auc           = -1.0
    best_model_weights = None

    # ── Loop de entrenamiento ─────────────────────────────────────────────────
    for epoch in tqdm(range(1, args.max_epoch + 1), total=args.max_epoch, dynamic_ncols=True):

        loss_dict = train_v2(
            train_loader, model, optimizer, device,
            triplet_weight=args.triplet_weight,
            tau=args.tau,
            scaler=scaler,
        )

        auc_val, ap_val = test_Over(val_loader, eval_model, val_args, device)

        if use_online_test:
            auc_test, ap_test = test_Over_online(backbone, processor, eval_model, args, device)
        else:
            auc_test, ap_test = auc_val, ap_val

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        hist["epoch"].append(epoch)
        hist["loss_total"].append(loss_dict["total"])
        hist["loss_bce"].append(loss_dict["bce"])
        hist["loss_triplet"].append(loss_dict["triplet"])
        hist["auc_val"].append(auc_val)
        hist["ap_val"].append(ap_val)
        hist["auc_test"].append(auc_test)
        hist["ap_test"].append(ap_test)
        hist["lr"].append(lr)

        print(
            f"\nEpoch {epoch}/{args.max_epoch} | LR: {lr:.2e} | "
            f"loss={loss_dict['total']:.4f} "
            f"(bce={loss_dict['bce']:.4f} tri={loss_dict['triplet']:.4f}) | "
            f"auc_val={auc_val:.4f} | auc_test={auc_test:.4f} | "
            f"ap_val={ap_val:.4f} | ap_test={ap_test:.4f}\n"
        )

        # ── Checkpoint ───────────────────────────────────────────────────────
        if auc_test > best_auc:
            best_auc           = auc_test
            best_model_weights = {k: v.cpu().clone() for k, v in _model_core.state_dict().items()}
            ckpt_path = os.path.join(
                args.ckpt_dir,
                f"vmae_v2_{auc_test:.4f}_ep{epoch}.pt",
            )
            torch.save(best_model_weights, ckpt_path)
            print(f"  [ckpt] Guardado: {ckpt_path}")

        # ── Visualización por video (cada 10 épocas y la última) ─────────────
        if (epoch % 10 == 0 or epoch == args.max_epoch) and args.gt_annotation:
            save_epoch_viz(
                model               = eval_model,
                val_loader          = val_loader,
                gt_annotation_path  = args.gt_annotation,
                output_dir          = args.output_dir / "viz",
                epoch               = epoch,
                device              = device,
                clip_step           = getattr(args, "clip_step", 16),
            )

        # ── Plot ──────────────────────────────────────────────────────────────
        if epoch % 5 == 0 or epoch == args.max_epoch:
            plot_training_curves(
                hist["epoch"],
                hist["loss_total"], hist["loss_bce"], hist["loss_triplet"],
                hist["auc_val"],   hist["auc_test"],
                hist["ap_val"],    hist["ap_test"],
                hist["lr"],
                save_path=os.path.join(args.ckpt_dir, "training_curves_v2.png"),
            )

        clear_gpu_cache()

    print(f"\n[main] Entrenamiento completo. Mejor AUC test: {best_auc:.4f}")
