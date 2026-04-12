# VideoMAE VAD Classifier v2 — Metric Learning

Refactorización del clasificador de anomalías con **Soft Triplet Loss + OHNM** para superar el estancamiento de AUC observado en v1.

---

## Motivación

El clasificador v1 usa solo BCE. El problema: BCE empuja scores hacia 0 o 1, pero no impone ninguna estructura en el espacio de embeddings. Dos clips de *Fighting* pueden tener representaciones internas completamente distintas y el modelo aún funciona si su score final es correcto.

Esto causa estancamiento: el modelo aprende a discriminar bien los casos fáciles pero no generaliza en clips ruidosos o ambiguos (problema real en UCF-Crime con pseudo-labels).

La v2 añade **Metric Learning**: fuerza que las anomalías estén cerca entre sí y lejos de los normales en un espacio de distancias de 128 dimensiones, lo que mejora la estructura interna y la generalización.

---

## Arquitectura

```
Input: (B, 768) float32
  │  features VideoMAE pre-extraídas — backbone congelado en Fase 1
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  SharedEncoder                                              │
│  fc1(768→512) + fc_att1(Softmax) → residual → ReLU → Drop  │
│                                                             │
│  ← recibe ∇BCE  +  ∇Soft Triplet  (ambos gradientes)       │
└──────────────────────────┬──────────────────────────────────┘
                           │ (B, 512)
              ┌────────────┴─────────────┐
              ▼                          ▼
┌─────────────────────────┐   ┌──────────────────────────────┐
│  ClassificationHead     │   │  ProjectionHead              │
│  fc2(512→32)+att2       │   │  Linear(512→256)             │
│  → ReLU → Drop          │   │  → BatchNorm1d               │
│  → fc3(32→1) → Sigmoid  │   │  → ReLU                      │
│                         │   │  → Linear(256→128)           │
│  score (B,1) ∈ [0,1]    │   │  → F.normalize(p=2)          │
│  ← solo ∇BCE            │   │                              │
└────────────┬────────────┘   │  embedding (B,128) ‖e‖₂=1   │
             │                │  ← solo ∇Soft Triplet        │
             ▼                └──────────────┬───────────────┘
         BCE Loss                            ▼
                                    Soft Triplet Loss
                                         │
                          L_total = L_bce + λ_t · L_triplet
```

### Por qué el SharedEncoder recibe ambos gradientes

Es el diseño clave. Al bifurcar desde los 512-dim internos (no desde el input de 768):

- La **BCE** entrena el SharedEncoder para ser discriminativa
- El **Soft Triplet** entrena el SharedEncoder para tener estructura métrica
- Los dos gradientes se suman en `fc1/att1` — el encoder aprende ambas cosas a la vez

Si el ProjectionHead fuera paralelo desde el input 768, los dos losses serían independientes y no habría aprendizaje conjunto.

---

## Loss combinada

### BCE con Label Smoothing

```python
target_smooth = label * (1 - ε) + ε / 2    # ε = 0.1
# label=1 → target=0.95
# label=0 → target=0.05
```

Necesario porque ~15-20% de las pseudo-labels de UCF-Crime son incorrectas. Evita que el modelo sea excesivamente confiado en etiquetas ruidosas.

### Soft Triplet Loss con OHNM

```
L_soft = log(1 + exp( (d_ap² − d_an²) / τ ))
```

**Triplets:**
- **Anchor** = clip anómalo del batch
- **Hard Positive** = anomalía más **lejana** al anchor en el batch (fuerza compacidad)
- **Hard Negative** = normal más **cercana** al anchor (fuerza separación)

**Por qué Soft y no Triplet estándar:**

```
Triplet estándar:  max(0,  d_ap² − d_an² + α)
                        ↑
                   zona muerta: gradiente=0 cuando d_an−d_ap > α
                   Con ruido en UCF-Crime, muchos pares se "resuelven"
                   prematuramente y el modelo deja de aprender de ellos.

Soft Triplet:  log(1 + exp( (d_ap² − d_an²) / τ ))
                   ↑
               siempre hay gradiente — incluso en pares ya bien separados
               el modelo sigue mejorando la estructura interna.
```

**Temperatura τ:**

| τ | Comportamiento |
|---|---|
| 0.1–0.3 | Gradiente concentrado en hard cases. Agresivo, puede inestabilizar con mucho ruido |
| 0.5 | Equilibrio recomendado para UCF-Crime pseudo-labels |
| 0.7–1.0 | Gradiente distribuido. Más robusto a ruido, convergencia más lenta |

**OHNM no requiere cambiar el DataLoader.** El `BalancedBatchSampler` ya garantiza 50% anomalía + 50% normal. El mining se hace dentro de cada batch en tiempo real. Si un batch no tiene ≥2 anomalías + ≥1 normal, el Triplet devuelve 0 (skip graceful).

**Distancias en hiperesfera:** `F.normalize` fuerza `‖e‖₂=1`. Todas las distancias quedan en `[0, 4]`, lo que hace el loss estable y evita soluciones triviales.

---

## Scheduler: warmup lineal → CosineAnnealing

```
Épocas 1..warmup:   lr = lr_base × linspace(0.1 → 1.0)
Épocas warmup..end: lr = eta_min + ½(lr_base−eta_min)(1 + cos(πt/T))
```

El warmup es importante porque el Soft Triplet tiene gradiente desde el primer batch. Sin warmup, los primeros updates del ProjectionHead pueden desestabilizar el SharedEncoder antes de que la BCE haya establecido una representación inicial útil.

---

## Archivos

| Archivo | Responsabilidad |
|---|---|
| `model_v2.py` | `VideoMAE_VAD_Classifier_v2` — SharedEncoder + ClassificationHead + ProjectionHead |
| `train_v2.py` | `LabelSmoothingBCE` + `soft_triplet_ohnm` + `train_v2()` |
| `option_v2.py` | Argumentos CLI (hereda paths de v1, añade hiperparámetros métricos) |
| `mainVideoMAE_v2.py` | Entry-point — scheduler, loop, checkpoints, plots |

Reutiliza de `videoMae_Freeze/` (v1) sin copiar:
- `dataset.py` — `Dataset_VideoMAE`, `BalancedBatchSampler`
- `test.py` — `test_Over`, `test_Over_online`
- `gpu_utils.py` — `get_optimal_device`, `try_init_rmm_pool`

---

## Uso

```bash
cd /home/jonathan/TransLow_Distill

uv run videoMae_Freeze_v2/mainVideoMAE_v2.py \
  --features_dir        data/vmae_features \
  --clip_names_path     data/pseudo_labels/clip_names.npy \
  --gt_binary_path      data/pseudo_labels/gt_binary.npy \
  --gt_annotation       resources/Temporal_Anomaly_Annotation.txt \
  --test_video_root     /mnt/gpu02/datasets/Data_Arco_V2/UCF-Crime/videos \
  --ckpt_dir            data/eval_results/vmae_v2_ckpt \
  --output_dir          data/eval_results/vmae_v2_output \
  --stats_dir           data/eval_results/vmae_v2_stats \
  --max_epoch           100 \
  --batch_size          64 \
  --triplet_weight      0.3 \
  --tau                 0.5 \
  --warmup_epochs       5
```

### Deshabilitar Soft Triplet (equivale a v1)

```bash
--triplet_weight 0.0
```

---

## Hiperparámetros nuevos

| Arg | Default | Rango sugerido | Descripción |
|---|---|---|---|
| `--proj_hidden` | 256 | 128–512 | Dim oculta del ProjectionHead |
| `--proj_out` | 128 | 64–256 | Dim del embedding de salida |
| `--triplet_weight` | 0.3 | 0.1–0.5 | λ_t: peso del Soft Triplet |
| `--tau` | 0.5 | 0.2–1.0 | Temperatura del Soft Triplet |
| `--label_smoothing` | 0.1 | 0.05–0.15 | Suavizado BCE |
| `--warmup_epochs` | 5 | 3–10 | Épocas de warmup lineal |
| `--eta_min` | 1e-6 | 1e-7–1e-5 | LR mínimo CosineAnnealing |

---

## Roadmap de fases

### Fase 1 — actual
Backbone congelado. SharedEncoder + ambas cabezas entrenadas desde cero.

### Fase 2
Descongelar última capa VideoMAE + añadir convolución temporal. El ProjectionHead se reconecta al nuevo output. La loss no cambia.

### Fase 3
Añadir Guided Attention:
```
L_total = L_bce + λ_t · L_triplet + λ_att · L_attention
```

Los pesos de ProjectionHead y los hiperparámetros `τ`, `λ_t` se mantienen entre fases.
