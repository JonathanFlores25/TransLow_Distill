"""
core/base_expert.py
===================
Contrato base (ABC) para todos los Expertos de Arconte.

Filosofía de diseño:
    - La clase base NO impone cómo funciona el experto internamente.
      Cada experto puede usar 5 frames con Top-K o 64 frames con stride; eso
      es un detalle de implementación privado.

    - Lo que SÍ se impone es el "contrato público" que el Orquestador (main.py)
      necesita para operar de forma uniforme sobre cualquier lista de expertos:
        1.  load()              → inicializar CLIP y lanzar el worker thread.
        2.  process_heuristics()→ disparador espacial/temporal, llamado cada frame.
        3.  predict()           → inferencia CLIP pura, llamada por el worker interno.
        4.  is_active           → propiedad de lectura del estado de confirmación.
        5.  label               → nombre de la anomalía ("PELEA", "FUEGO", etc.)
        6.  get_display_data()  → datos opcionales para el dibujado en pantalla.

Formato de entrada de predict():
    List[np.ndarray]  →  lista de crops BGR (sin normalizar), tamaño libre.
    Cada elemento es un frame recortado que el experto ya preprocesó.

Formato de salida de predict():
    dict que SIEMPRE contiene al menos:
        {
            "detected": bool,   # True si el experto confirma la anomalía
            "score":    float,  # Puntuación de confianza principal (0.0 … ∞)
        }
    Puede incluir campos adicionales (hits, per_frame, raw, etc.) libremente.

Formato de tracker_data en process_heuristics():
    {
        "persons_xyxy":  np.ndarray  shape [N, 4]  – coords [x1,y1,x2,y2]
        "persons_ids":   np.ndarray  shape [N]     – IDs ByteTrack
        "vehicles_xyxy": np.ndarray  shape [M, 4]
        "vehicles_ids":  np.ndarray  shape [M]
        "vehicles_confs":np.ndarray  shape [M]     – confianza YOLO de cada veh.
        "frame_idx":     int                       – índice global del frame
    }
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Tipo de retorno canónico de predict()
# ---------------------------------------------------------------------------
# No es obligatorio usar este TypeAlias (Python 3.9 compatible), pero
# documenta exactamente qué se espera del dict de salida.
ExpertResult = Dict[str, Any]
# Ejemplo mínimo: {"detected": True, "score": 4.9}
# Ejemplo extendido Fight: {"detected": True, "score": 4.9, "hits": 6, "raw": 4.5}
# Ejemplo extendido Crash: {"detected": True, "score": 0.93, "hits": 3, "per_frame": [...]}


class BaseExpert(ABC):
    """
    Clase base abstracta para los Expertos de Arconte.

    Todos los expertos (FightExpert, CrashExpert, FireExpert, SmokeExpert)
    deben heredar de esta clase e implementar los métodos abstractos.

    Ciclo de vida típico en el Orquestador:
    ─────────────────────────────────────
    expert = MiExperto()
    expert.load(clip_model, clip_preprocess)   # una vez, antes del bucle

    while cap.isOpened():
        tracker_data = tracker.process(frame)
        expert.process_heuristics(frame, tracker_data)   # cada frame

        if expert.is_active:
            anomalias.add(expert.label)
            extra = expert.get_display_data()            # para dibujar bboxes, etc.
    """

    # ------------------------------------------------------------------
    # Métodos abstractos obligatorios
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self, model: Any, preprocess: Any) -> None:
        """
        Carga el modelo CLIP compartido, codifica los prompts de texto del
        experto y lanza el worker thread de inferencia en segundo plano.

        Parámetros:
            model      : Objeto CLIP ya cargado (clip.load() en main.py).
            preprocess : Función de preprocesamiento de imágenes de CLIP.

        Notas:
            - model y preprocess son compartidos por todos los expertos para
              evitar duplicar memoria GPU.
            - Este método DEBE llamarse exactamente una vez antes de iniciar
              el bucle principal.
            - La implementación típica termina con el lanzamiento del thread:
                threading.Thread(target=self._worker_loop, daemon=True).start()
        """
        ...

    @abstractmethod
    def process_heuristics(
        self,
        frame: np.ndarray,
        tracker_data: Dict[str, Any],
    ) -> None:
        """
        Disparador espacial/temporal. Se invoca en CADA frame del bucle principal.

        Responsabilidades de la implementación:
            1. Evaluar la condición heurística propia del experto
               (IoU de personas, cercanía de vehículos, máscara HSV de fuego…).
            2. Gestionar el buffer interno de crops.
            3. Enviar al worker CLIP via queue interna cuando el buffer esté lleno.
            4. Decrementar TTL o contadores de confirmación si corresponde.

        Parámetros:
            frame        : Frame BGR de tamaño original (sin escalar).
            tracker_data : Salidas del tracker. Claves disponibles:
                "persons_xyxy"   np.ndarray [N,4]
                "persons_ids"    np.ndarray [N]
                "vehicles_xyxy"  np.ndarray [M,4]
                "vehicles_ids"   np.ndarray [M]
                "vehicles_confs" np.ndarray [M]
                "frame_idx"      int

        Retorna:
            None. El estado de detección se actualiza internamente y se
            expone a través de la propiedad `is_active`.
        """
        ...

    @abstractmethod
    def predict(self, frames: List[np.ndarray]) -> ExpertResult:
        """
        Inferencia CLIP pura sobre un batch de crops preprocesados.
        Es llamado exclusivamente por el worker thread interno del experto.

        Parámetros:
            frames : Lista de imágenes BGR (np.ndarray), una por slot del buffer.
                     El tamaño puede variar según el experto (20 para pelea/fuego,
                     64 para choques). El experto aplica su propio stride interno.

        Retorna:
            ExpertResult (dict) que SIEMPRE incluye:
                "detected" : bool   – True si la anomalía se confirma en este batch.
                "score"    : float  – Puntuación principal de confianza.
            Puede incluir campos adicionales libremente (hits, per_frame, raw…).
        """
        ...

    # ------------------------------------------------------------------
    # Propiedades abstractas: estado público para el Orquestador
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """
        True si el experto tiene una detección confirmada en el frame actual.
        El Orquestador lee esta propiedad cada frame para construir el set
        de anomalías activas.
        """
        ...

    @property
    @abstractmethod
    def label(self) -> str:
        """
        Etiqueta de texto de la anomalía que produce este experto.
        Ejemplos: "PELEA", "CHOQUE", "FUEGO", "HUMO".
        El Orquestador la usa para el log de transiciones y el overlay.
        """
        ...

    # ------------------------------------------------------------------
    # Métodos con implementación por defecto (opcionales de sobreescribir)
    # ------------------------------------------------------------------

    def get_display_data(self) -> Dict[str, Any]:
        """
        Datos adicionales opcionales para el sistema de dibujado en pantalla.

        La implementación por defecto retorna un dict vacío.
        Los expertos que quieran exponer información visual (p.ej. FireExpert
        expone las bounding boxes de las llamas) deben sobreescribir este método.

        Retorna un dict con cualquier subconjunto de:
            {
                "bboxes":     List[np.ndarray]  – bboxes a dibujar [[x1,y1,x2,y2], ...]
                "last_crop":  np.ndarray        – último crop BGR para mini-preview
                "color":      Tuple[int,int,int]– color BGR del overlay (por defecto rojo)
                "extra_text": str               – línea de texto adicional en el HUD
            }
        """
        return {}

    # ------------------------------------------------------------------
    # Método utilitario concreto: representación legible
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "ACTIVO" if self.is_active else "en espera"
        return f"<{self.__class__.__name__} label='{self.label}' estado={status}>"
