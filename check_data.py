import numpy as np
import cv2
import os
from pathlib import Path

# Carga un clip aleatorio de los que acabas de generar
data_path = Path("data/processed/normal/Normal_Videos084_x264_s000176.npz")
data = np.load(data_path)

video = data['video']       # (16, 3, 224, 224)
mask = data['attn_mask']    # (224, 224)
label = data['label']

print(f"Clip: {data_path.name} | Label: {label}")

# Tomamos el frame central del clip (el 8)
# Revertimos CHW a HWC y normalización para visualizar
frame = video[8].transpose(1, 2, 0) * 255
frame = frame.astype(np.uint8)
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Aplicar mapa de color a la máscara de atención
heatmap = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

# Guardar para inspección
cv2.imwrite("test_attention_check.png", overlay)
print("Imagen de control guardada como 'test_attention_check.png'")