---
name: VideoMAE VAD v2 — Arquitectura y Teoría
description: Arquitectura completa de videoMae_Freeze_v2, definiciones de cada componente, teoría del Soft Triplet Loss y roadmap de fases
type: project
---

## VideoMAE VAD Classifier v2 — Metric Learning Edition

Ubicación: `videoMae_Freeze_v2/`
Archivos: `model_v2.py`, `train_v2.py`, `option_v2.py`, `mainVideoMAE_v2.py`

---

## Motivación

El clasificador v1 estancaba el AUC porque BCE sola no fuerza estructura en el
espacio de embeddings — solo empuja scores hacia 0 o 1. La v2 añade Metric
Learning para que las anomalías estén cerca entre sí y lejos de los normales
en un espacio de distancias, lo que mejora la generalización en clips ruidosos.

---

## Arquitectura del modelo (`model_v2.py`)

```
Input: (B, 768) float32  ← features VideoMAE pre-extraídas (congeladas en Fase 1)
  │
  ▼
[SharedEncoder]
  fc1(768→512) + fc_att1(768→512, Softmax) → residual → ReLU
  Salida: (B, 512)  ← SIN dropout aquí (ver bug fix abajo)
  ← recibe ∇BCE  +  ∇Soft Triplet  (ambos gradientes se suman aquí)
  │
  ├──────────────────────────────────────────────────────────┐
  ▼                                                          ▼
[ClassificationHead]                               [ProjectionHead]
  Dropout(0.6) → fc2(512→32) + fc_att2 → residual    Linear(512→256)
  → ReLU → Dropout(0.6) → fc3(32→1) → Sigmoid        → BatchNorm1d(256)
  Salida: (B, 1)  score ∈ [0,1]                       → ReLU
  ← recibe solo ∇BCE                                  → Linear(256→128)
                                                       → F.normalize(p=2)
                                                       Salida: (B, 128) ‖e‖₂=1
                                                       ← recibe solo ∇Soft Triplet
```

### BUG CRÍTICO CORREGIDO (sesión 2)

**Problema**: `_encode` aplicaba `Dropout(0.6)` ANTES de ramificar. El
ProjectionHead recibía features que cambiaban 60% por batch → embeddings
inconsistentes → metric collapse → triplet loss se fijaba en ln(2)=0.693
desde las primeras épocas, sin gradiente útil.

**Fix**: Dropout movido al inicio de `_classify`. El ProjectionHead recibe
features limpias. La clasificación mantiene el mismo número total de dropouts.

**Fix 2**: `auc_val=0.5000` siempre — `test_Over` buscaba stems de val (train)
en la GT de test → 0 matches → AUC trivial. Solución: `val_args.gt_annotation = None`
en mainVideoMAE_v2.py → usa clip-level metrics con pseudo-labels.

### API del modelo

- `model(x)` → `(score, embedding)` — usado en entrenamiento
- `model.predict(x)` → `score` — usado en evaluación (compatible con test.py v1)
- Parámetros entrenables: ~985,057

### Por qué el BatchNorm en el ProjectionHead

Con pseudo-labels ruidosas de UCF-Crime, las features de clips mal etiquetados
crean gradientes erráticos. El BN normaliza las activaciones por batch,
estabilizando el entrenamiento y evitando que unos pocos clips ruidosos
dominen el espacio de embeddings.

---

## Loss combinada (`train_v2.py`)

```
L_total = L_bce_smooth  +  λ_t · L_soft_triplet
```

### 1. BCE con Label Smoothing (ε=0.1)

```
target_smooth = label × (1−ε) + ε/2
label=1 → target=0.95   label=0 → target=0.05
```

Evita que el modelo sea excesivamente confiado en pseudo-labels que pueden
estar mal etiquetadas (~15-20% de ruido en UCF-Crime).

### 2. Soft Triplet Loss con OHNM

```
L_soft = log(1 + exp( (d_ap² − d_an²) / τ ))
```

**Componentes:**
- `d_ap²` = distancia euclidiana² entre Anchor y Hard Positive
  - Hard Positive = anomalía más LEJANA al anchor en el batch
- `d_an²` = distancia euclidiana² entre Anchor y Hard Negative
  - Hard Negative = normal más CERCANA al anchor en el batch
- `τ` = temperatura (default 0.5)

**Por qué Soft Triplet y no Triplet estándar:**

Triplet estándar: `max(0, d_ap² − d_an² + α)`
→ Zona muerta: cuando d_an−d_ap > α, gradiente = 0
→ Con pseudo-labels ruidosas, muchos pares se "resuelven" prematuramente
   y el modelo deja de aprender de ellos aunque estén mal aprendidos.

Soft Triplet: `log(1 + exp(...))`
→ Sin zona muerta: siempre hay gradiente, incluso en pares ya resueltos
→ τ bajo (0.1–0.3): gradiente concentrado en hard cases (agresivo)
→ τ alto (0.7–1.0): gradiente distribuido, más robusto a ruido
→ Compatible con descongele futuro del backbone (gradiente fluye sin cortes)

**OHNM (Online Hard Negative Mining):**
- No se necesita cambiar el DataLoader (sigue devolviendo pares features/label)
- BalancedBatchSampler ya garantiza 50% anomalía + 50% normal por batch
- El mining se hace dentro de cada batch en tiempo real
- Si un batch no tiene ≥2 anomalías + ≥1 normal → skip graceful (loss=0)

**Distancias en hiperesfera unitaria:**
- F.normalize fuerza ‖e‖₂=1 → embeddings en S^127
- d² = 2 − 2·cos(θ) ∈ [0, 4] — siempre acotado
- Evita soluciones triviales (embeddings a ±∞)

---

## Scheduler: LinearLR (warmup) → CosineAnnealingLR

```
Épocas 1..warmup_epochs:  lr = lr_base × linspace(0.1 → 1.0)
Épocas warmup..max_epoch: lr = eta_min + 0.5×(lr_base−eta_min)×(1+cos(π·t/T))
```

Defaults: warmup_epochs=5, eta_min=1e-6
El warmup estabiliza el encoder compartido antes de aplicar la presión métrica
del Soft Triplet (que tiene gradiente desde el primer batch).

---

## Flujo de gradientes (clave del diseño)

```
L_total.backward()
    ↓
∇L_bce    → ClassificationHead → SharedEncoder (fc1+att1)
∇L_triplet → ProjectionHead   → SharedEncoder (fc1+att1)

En SharedEncoder los dos gradientes se SUMAN.
El encoder aprende a producir representaciones que:
  · Son discriminativas para clasificación binaria (BCE)
  · Mantienen estructura métrica: anomalías juntas, lejos de normales (Triplet)
```

---

## Hiperparámetros nuevos (option_v2.py)

| Arg | Default | Descripción |
|---|---|---|
| `--proj_hidden` | 256 | Dim oculta del ProjectionHead |
| `--proj_out` | 128 | Dim del embedding de salida |
| `--triplet_weight` | 0.3 | λ_t: peso del Soft Triplet |
| `--tau` | 0.5 | Temperatura del Soft Triplet |
| `--label_smoothing` | 0.1 | Suavizado BCE |
| `--warmup_epochs` | 5 | Épocas de warmup lineal |
| `--eta_min` | 1e-6 | LR mínimo del CosineAnnealing |

---

## Roadmap de fases

### Fase 1 (actual — videoMae_Freeze_v2/)
- VideoMAE backbone completamente congelado
- FC Classifier + Projection Head entrenados desde cero
- Loss: BCE smooth + Soft Triplet OHNM
- Objetivo: aprender representación métrica sobre pseudo-labels

### Fase 2 (próxima — videoMae_Freeze_v3/)
- Extrae tokens ESPACIALES del backbone: (8, 14, 14, 768) float16 por clip
  - T=8 (CLIP_LEN/tubelet_size = 16/2)
  - H=W=14 (OUTPUT_SIZE/patch_size = 224/16)
  - ~2.4 MB/clip, ~1.75 TB para todo UCF-Crime
- Arquitectura: Factored Attention (TimeSformer style)
  - Atención espacial sobre 196 tokens (14×14) por timestep
  - Atención temporal sobre 8 timesteps
  - Supervisado por máscaras de atención RANDOM (control experiment)
- FE: `python FE_VideoMAE.py --spatial --batch-size 16 --output /disco/vmae_features_spatial`

### Fase 3 (futura)
- Igual que Fase 2 pero con atención GUIADA por Arconte teacher
  - Máscaras de Gaussianas 224×224 → downsample a 14×14
  - Loss total: BCE + Soft Triplet + λ_att · L_attention
- Hipótesis para paper: guided > random > baseline → knowing WHERE proves value

---

## Compatibilidad con v1

- `dataset.py`, `test.py`, `gpu_utils.py` de v1 se importan directamente (sin copiar)
- `test_Over` y `test_Over_online` usan `model.predict(x)` via `_PredictWrapper`
- Los checkpoints v1 y v2 son incompatibles (state_dict diferente)
- `--triplet_weight 0.0` desactiva el Soft Triplet → equivale funcionalmente a v1

**Why:** Jonathan quiere superar el estancamiento de AUC con Metric Learning,
manteniendo compatibilidad con la infraestructura de evaluación existente y
dejando la arquitectura lista para descongelar el backbone en fases futuras.
