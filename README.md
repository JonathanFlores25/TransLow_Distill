# TransLow_Distill

Pipeline de Video Anomaly Detection (VAD) sobre UCF-Crime usando el framework Arconte como teacher y VideoMAE como backbone para el clasificador estudiante.

---

## Arquitectura general

```
UCF-Crime videos
      │
      ▼
[Arconte Experts]  ←── experts_V2_0_0/
      │  scores por clip (fight/crash/fire/robbery/carparts)
      ▼
[pseudo_label_generator.py]
      │  clip_names.npy + gt_binary.npy (pseudo-labels soft 0–1)
      ▼
[FE_VideoMAE.py]
      │  data/vmae_features/{category}/{stem}_s{start}.npy  (768-dim)
      ▼
[videoMae_Freeze/]  ←── clasificador entrenado con pseudo-labels
      │  AUC/AP frame-level contra Temporal_Anomaly_Annotation.txt
      ▼
  Métricas VAD  ←── src/metrics_library.py
```

---

## Estructura del repositorio

```
TransLow_Distill/
├── experts_V2_0_0/          # Expertos Arconte (teacher)
│   ├── fight_expert.py
│   ├── crash_expert.py
│   ├── fire_expert.py
│   ├── robbery_expert.py
│   └── carparts_expert.py
│
├── core/                    # Clases base compartidas
│   ├── base_expert.py       # BaseExpert — interfaz común de todos los expertos
│   └── tracker.py           # ArconteTracker — seguimiento de objetos
│
├── videoMae_Freeze/         # Clasificador VideoMAE (Step actual)
│   ├── mainVideoMAE.py      # Entry-point de entrenamiento
│   ├── model.py             # VideoMAE_VAD_Classifier (FC + Softmax attention)
│   ├── dataset.py           # Dataset_VideoMAE + BalancedBatchSampler
│   ├── train.py             # concatenated_train_feedback
│   ├── test.py              # test_Over (val) + test_Over_online (test sin disco)
│   ├── option.py            # Argumentos CLI
│   └── gpu_utils.py         # Utilidades GPU con fallback automático
│
├── src/
│   ├── metrics_library.py   # Métricas VAD frame-level (AUC, AP, mAA)
│   └── inference_arconte.py # Inferencia con pipeline Arconte completo
│
├── pseudo_label_generator.py  # Genera pseudo-labels combinando Arconte + VideoMAE
├── FE_VideoMAE.py             # Extrae features VideoMAE (768-dim) a disco
├── main_V2_0_0.py             # Orquestador Arconte V2 (inferencia completa)
├── data_processor_GPU_{1-4}.py# Procesamiento distribuido en 4 GPUs
│
├── resources/
│   ├── Temporal_Anomaly_Annotation.txt  # GT frame-level test set (290 videos)
│   ├── Anomaly_Train.txt                # Lista de videos de entrenamiento
│   └── Anomaly_Train_GPU_{1-4}.txt      # Splits por GPU
│
└── data/
    ├── vmae_features/         # Features VideoMAE train (plano, sin split)
    ├── vmae_features_test/    # Features VideoMAE test (generadas en época 1)
    └── pseudo_labels/
        ├── clip_names.npy     # (N,) nombres de clips anómalos
        └── gt_binary.npy      # (N,) soft labels 0.0–1.0
```

---

## Módulos principales

### `experts_V2_0_0/` — Expertos Arconte (teacher)

Cada experto recibe un clip de video y devuelve `(score, is_active)`:

| Expert | Categorías UCF-Crime |
|---|---|
| `FightExpert` | Fighting, Assault, Abuse, Arrest, Shooting |
| `CrashExpert` | RoadAccidents, Accident |
| `FireExpert` | Arson, Explosion |
| `RobberyExpert` | Burglary, Robbery, Vandalism |
| `CarPartsExpert` | Stealing, Shoplifting |

### `pseudo_label_generator.py` — Generación de pseudo-labels

Combina scores Arconte con features VideoMAE para generar labels de entrenamiento:

1. **Expert Scoring** — corre Arconte sobre todos los clips del train set
2. **Distribución normal** — ajusta Gaussiana sobre clips normales (norma L2 de features VideoMAE)
3. **GT Generation** — sliding window 20% + intersección con expert activation → label soft 0–1
4. **Visual clips** — guarda MP4 con overlay para verificación visual

Salida: `data/pseudo_labels/clip_names.npy` + `gt_binary.npy`

```bash
python pseudo_label_generator.py \
  --video-root /mnt/gpu02/datasets/Data_Arco_V2/UCF-Crime/videos \
  --features-dir data/vmae_features \
  --output-dir data/pseudo_labels
```

### `FE_VideoMAE.py` — Extractor de features

Extrae embeddings `(768,)` por clip usando `MCG-NJU/videomae-base-finetuned-kinetics`.
Arquitectura: N reader threads (I/O) → queue → GPU batch processor.

- Parámetros de clip: `CLIP_LEN=16`, `STRIDE=2`, `CLIP_STEP=16`, `OUTPUT_SIZE=224`
- Salida: `data/vmae_features/{category}/{stem}_s{start:06d}.npy`

```bash
python FE_VideoMAE.py --workers 4 --batch-size 32
```

### `videoMae_Freeze/` — Clasificador (etapa de entrenamiento actual)

Clasificador binario entrenado sobre features VideoMAE pre-extraídas.

**Arquitectura** (`model.py` — `VideoMAE_VAD_Classifier`):
```
768 → fc1(768→512) + fc_att1(Softmax) residual → ReLU → Dropout(0.6)
512 → fc2(512→32)  + fc_att2(Softmax) residual → ReLU → Dropout(0.6)
32  → fc3(32→1) → Sigmoid
```
Idéntica a `Model_V3_Connection` de Step2_Detector, adaptada para 768-dim.

**Evaluación dual:**
- `test_Over` (val): carga `.npy` pre-extraídas — rápido, cada época
- `test_Over_online` (test): video crudo → VideoMAE backbone en línea → clasificador
  - Primera época: extrae y cachea features en `data/vmae_features_test/`
  - Épocas siguientes: carga de caché (igual de rápido que val)

**Comando de entrenamiento:**
```bash
cd /home/jonathan/TransLow_Distill

uv run videoMae_Freeze/mainVideoMAE.py \
  --features_dir data/vmae_features \
  --clip_names_path data/pseudo_labels/clip_names.npy \
  --gt_binary_path data/pseudo_labels/gt_binary.npy \
  --gt_annotation resources/Temporal_Anomaly_Annotation.txt \
  --test_video_root /mnt/gpu02/datasets/Data_Arco_V2/UCF-Crime/videos \
  --ckpt_dir data/eval_results/videomae_ckpt \
  --output_dir data/eval_results/videomae_output \
  --stats_dir data/eval_results/videomae_stats \
  --max_epoch 100 \
  --batch_size 32 \
  --num_workers 4
```

### `src/metrics_library.py` — Librería de métricas

Métricas VAD model-agnostic. Usar siempre para evaluación frame-level.

Funciones principales:
- `parse_gt_annotations(txt_path)` — parsea `Temporal_Anomaly_Annotation.txt`
- `clips_to_frames(clip_starts, scores, active, total_frames)` — expande clip → frame
- `build_gt_frame_array(annotation, total_frames)` — GT binario por video
- `frame_auc(scores, gt)` / `frame_ap(scores, gt)` — métricas frame-level
- `compute_all_metrics(results, annotations)` — suite completa (AUC, AP, mAA)

---

## Dataset UCF-Crime

- **Train**: 1,610 videos con anotación de categoría (`resources/Anomaly_Train.txt`)
- **Test**: 290 videos con GT temporal frame-level (`resources/Temporal_Anomaly_Annotation.txt`)
- **Clips**: `CLIP_LEN=16` frames, `STRIDE=2`, `CLIP_STEP=16` → nombrados `{stem}_s{start:06d}`

### Rutas en servidor GPU02
```
Videos:   /mnt/gpu02/datasets/Data_Arco_V2/UCF-Crime/videos/{Category}/{stem}.mp4
Features: /home/jonathan/TransLow_Distill/data/vmae_features/{category}/
```

### Checkpoints de modelos base
```
.checkpoints/ViT-L-14.pt          # CLIP ViT-L/14
.checkpoints/ViT-B-16-32-f.pt     # ActionCLIP
.checkpoints/yolo11x.pt           # YOLO v11x (detección de objetos)
```

---

## Instalación

```bash
# Requiere uv
uv sync

# O con pip
pip install -r videoMae_Freeze/requirements.txt
```

Python 3.14+ recomendado (ver `.python-version`).
