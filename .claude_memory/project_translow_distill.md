---
name: TransLow_Distill — Descripción del proyecto
description: Pipeline VAD distillation, UCF-Crime, experts, rutas GPU/local, archivos clave, estado actual
type: project
originSessionId: eec95db5-34d2-4ecf-9efc-40e0b839e51a
---
## Proyecto: TransLow_Distill

Pipeline de **knowledge distillation** para Video Anomaly Detection (VAD) sobre UCF-Crime.
Tiene dos etapas principales activas:

---

### Etapa activa: videoMae_Freeze/

Clasificador binario de anomalías entrenado sobre features VideoMAE (768-dim).
Basado en la misma arquitectura que `Step2_Detector` (ArconteDetection), adaptada para VideoMAE.

#### Arquitectura del modelo (`videoMae_Freeze/model.py`)
Clase: `VideoMAE_VAD_Classifier` — idéntica a `Model_V3_Connection` de Step2 con input 768 en vez de 192:
```
768 → fc1(768→512) + fc_att1(768→512, Softmax) residual → ReLU → Dropout(0.6)
512 → fc2(512→32)  + fc_att2(512→32,  Softmax) residual → ReLU → Dropout(0.6)
32  → fc3(32→1) → Sigmoid
```
Loss: `BCELoss`. Optimizer: SGD lr=1e-4, momentum=0.9, nesterov=True, wd=5e-4.

#### Features de entrenamiento
- Ubicación: `data/vmae_features/{category}/{stem}_s{start:06d}.npy`
- Estructura plana (sin subdirs train/val/test) — el Dataset hace fallback automático
- 732,184 clips totales | 800 videos normales + 616 anomalía
- Pseudo-labels: `data/pseudo_labels/clip_names.npy` + `gt_binary.npy`
- Después de aplicar pseudo-labels: ~20,557 positivos (2.8%)

#### Evaluación
- **Val** cada época: `test_Over()` — features .npy pre-extraídas (rápido)
- **Test** cada época: `test_Over_online()` — video crudo → VideoMAE backbone en línea → clasificador
  - Primera pasada: extrae features y las cachea en `data/vmae_features_test/{Category}/`
  - Pasadas siguientes: carga de caché (rápido, sin backbone)
  - GT temporal: `resources/Temporal_Anomaly_Annotation.txt` (290 videos oficiales)
  - Videos test en: `/mnt/gpu02/datasets/Data_Arco_V2/UCF-Crime/videos`

#### Archivos clave de videoMae_Freeze/
- `model.py` — VideoMAE_VAD_Classifier
- `dataset.py` — Dataset_VideoMAE + BalancedBatchSampler
- `train.py` — concatenated_train_feedback (idéntico a Step2)
- `test.py` — test_Over (val, pre-extraídas) + test_Over_online (test, online con caché)
- `mainVideoMAE.py` — entry-point de entrenamiento
- `option.py` — argumentos CLI

#### Comando de entrenamiento (servidor GPU02)
```bash
cd /home/jonathan/TransLow_Distill
uv run videoMae_Freeze/mainVideoMAE.py --features_dir data/vmae_features --clip_names_path data/pseudo_labels/clip_names.npy --gt_binary_path data/pseudo_labels/gt_binary.npy --gt_annotation resources/Temporal_Anomaly_Annotation.txt --test_video_root /mnt/gpu02/datasets/Data_Arco_V2/UCF-Crime/videos --ckpt_dir data/eval_results/videomae_ckpt --output_dir data/eval_results/videomae_output --stats_dir data/eval_results/videomae_stats --max_epoch 100 --batch_size 32 --num_workers 4
```

#### Pendiente
- Agregar visualización por video: score frame-level sobre GT sombreado (`.png` por video en `output_dir/viz/`)

---

### Rutas importantes
- Videos UCF-Crime: `/mnt/gpu02/datasets/Data_Arco_V2/UCF-Crime/videos/{Category}/{stem}.mp4`
- GT temporal: `resources/Temporal_Anomaly_Annotation.txt` (290 videos test)
- Referencia Step2: `/home/jonathan/ArconteDetection/Step2_Detector/`
- metrics_library: `src/metrics_library.py` — usar siempre para métricas frame-level

### Parámetros clip
- `CLIP_LEN=16`, `STRIDE=2`, `CLIP_STEP=16`, `OUTPUT_SIZE=224`

### Rama activa
`JohnVersion` — Jonathan Flores. Colaborador: Sergio Huesca.

**Why:** El usuario necesita retomar trabajo entre sesiones sin reexplicar el proyecto desde cero.
**How to apply:** Leer este archivo al iniciar una conversación. Preguntar solo qué tarea específica se quiere continuar.
