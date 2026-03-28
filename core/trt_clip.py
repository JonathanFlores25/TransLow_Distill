"""
core/trt_clip.py
================
Wrappers TensorRT para CLIP ViT-L/14 y ActionCLIP ViT-B/16.

Clases:
    TRTCLIPVisual   — ejecuta el visual encoder TRT con batch dinamico.
    CLIPModelTRT    — drop-in replacement para el modelo CLIP compartido.
    ActionCLIPModelTRT — wrapper TRT para ActionCLIP (visual + fusion).
"""

import threading

import torch
import tensorrt as trt

_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ---------------------------------------------------------------------------
# TRTCLIPVisual — visual encoder generico sobre TensorRT
# ---------------------------------------------------------------------------

class TRTCLIPVisual:
    """
    Carga un .engine exportado del visual encoder de CLIP y ejecuta
    inferencia con batch dinamico usando buffers pre-alocados en GPU.

    Uso:
        visual = TRTCLIPVisual("model_visual_fp16.engine", max_batch=32)
        features = visual(images_tensor)  # [B, 3, 224, 224] -> [B, D]
    """

    def __init__(
        self,
        engine_path: str,
        input_name:  str = "image",
        output_name: str = "features",
        max_batch:   int = 32,
        img_size:    int = 224,
    ):
        self.input_name  = input_name
        self.output_name = output_name
        self.max_batch   = max_batch
        self.img_size    = img_size

        # Cargar engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(_TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Detectar dimension de salida del engine
        out_shape = self.engine.get_tensor_shape(self.output_name)
        self.feat_dim = out_shape[-1]

        # Pre-alocar buffers al tamano maximo
        self._input_buf  = torch.empty(
            (max_batch, 3, img_size, img_size),
            dtype=torch.float16, device="cuda",
        )
        self._output_buf = torch.empty(
            (max_batch, self.feat_dim),
            dtype=torch.float16, device="cuda",
        )

        # Stream CUDA dedicado
        self._stream = torch.cuda.Stream()
        self._lock = threading.Lock()

    def _run_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Ejecuta un batch que cabe en max_batch."""
        B = images.shape[0]
        with self._lock:
            with torch.cuda.stream(self._stream):
                self._input_buf[:B].copy_(images.half())
                self.context.set_input_shape(self.input_name, (B, 3, self.img_size, self.img_size))
                self.context.set_tensor_address(self.input_name,  self._input_buf.data_ptr())
                self.context.set_tensor_address(self.output_name, self._output_buf.data_ptr())
                self.context.execute_async_v3(self._stream.cuda_stream)
            self._stream.synchronize()
            return self._output_buf[:B].float().clone()

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W] float16 o float32 en GPU.
        Retorna: [B, D] float32 en GPU.
        """
        B = images.shape[0]
        if B <= self.max_batch:
            return self._run_batch(images)

        # Chunk para batches que exceden max_batch
        parts = []
        for i in range(0, B, self.max_batch):
            parts.append(self._run_batch(images[i:i + self.max_batch]))
        return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# CLIPModelTRT — drop-in para el modelo CLIP compartido
# ---------------------------------------------------------------------------

class CLIPModelTRT:
    """
    Wrapper que reemplaza el modelo CLIP original (openai/clip).
    - encode_image() usa TRTCLIPVisual (TensorRT FP16).
    - encode_text() delega al modelo PyTorch original (solo se llama en init).
    - Expone logit_scale del modelo original.
    """

    def __init__(self, original_model, engine_path: str, max_batch: int = 32):
        self._original   = original_model
        self._trt_visual = TRTCLIPVisual(engine_path, max_batch=max_batch)
        self.logit_scale = original_model.logit_scale

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self._trt_visual(images)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self._original.encode_text(text)

    def __getattr__(self, name):
        if name in ("_original", "_trt_visual", "logit_scale"):
            raise AttributeError(name)
        return getattr(self._original, name)


# ---------------------------------------------------------------------------
# ActionCLIPModelTRT — wrapper TRT para ActionCLIP (visual + fusion)
# ---------------------------------------------------------------------------

class ActionCLIPModelTRT:
    """
    Dos engines TRT encadenados para ActionCLIP:
      1. visual_engine: [T, 3, 224, 224] -> [T, D]
      2. fusion_engine: [1, T, D] -> [1, D]

    Interfaz compatible con _ActionCLIPVerifier.predict().
    """

    def __init__(
        self,
        visual_engine_path: str,
        fusion_engine_path: str,
        num_segments:       int = 32,
        feat_dim:           int = 512,
    ):
        self.num_segments = num_segments
        self.feat_dim     = feat_dim

        # Visual encoder TRT
        self._visual = TRTCLIPVisual(
            visual_engine_path,
            input_name="image",
            output_name="features",
            max_batch=num_segments,
            img_size=224,
        )

        # Fusion encoder TRT
        with open(fusion_engine_path, "rb") as f:
            runtime = trt.Runtime(_TRT_LOGGER)
            self._fusion_engine = runtime.deserialize_cuda_engine(f.read())

        self._fusion_context = self._fusion_engine.create_execution_context()

        # Buffers fusion
        self._fusion_input  = torch.empty(
            (1, num_segments, feat_dim), dtype=torch.float16, device="cuda"
        )
        self._fusion_output = torch.empty(
            (1, feat_dim), dtype=torch.float16, device="cuda"
        )

        self._stream = torch.cuda.Stream()

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [T, 3, 224, 224]
        Retorna: [1, D] float32
        """
        # Visual
        img_feat = self._visual(images)  # [T, D]

        # Fusion
        with torch.cuda.stream(self._stream):
            self._fusion_input[0].copy_(img_feat.half())

            self._fusion_context.set_input_shape("visual_features", (1, self.num_segments, self.feat_dim))
            self._fusion_context.set_tensor_address("visual_features", self._fusion_input.data_ptr())
            self._fusion_context.set_tensor_address("video_features",  self._fusion_output.data_ptr())

            self._fusion_context.execute_async_v3(self._stream.cuda_stream)

        self._stream.synchronize()
        return self._fusion_output.float()
