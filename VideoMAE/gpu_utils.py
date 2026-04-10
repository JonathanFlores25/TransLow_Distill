"""
gpu_utils.py
============
GPU device management for VideoMAE VAD Classifier (Step3).

Design rationale vs Step2_Detector:

- No module-level side effect calls: Step2/gpu_utils.py called
  check_gpu_availability() at the bottom of the module, which set global
  state at import time and could trigger CUDA initialization as a side
  effect inside DataLoader workers. Here all detection is deferred to
  get_optimal_device() which is called explicitly by the entry-point.

- get_optimal_device() logs device info directly: Step3 combines device
  detection and logging into a single call, removing the need for callers
  to separately call print_gpu_status(). This makes the entry-point
  startup code more concise.

- RMM pool init is fully guarded: any exception (not just ImportError) is
  silently handled so the pipeline runs on systems without RAPIDS installed.

- Removed cuDF / dataframe utilities: Step3 does not use dataframes — all
  data is .npy files processed by numpy/torch. Keeping those utilities
  would add dead code and a dependency on cuDF availability.

- VRAM reporting uses torch.cuda.mem_get_info() (available since PyTorch
  1.9) which gives the OS-level free/total VRAM on the selected device,
  more accurate than total - reserved for scheduling decisions.
"""

from __future__ import annotations

import logging
import warnings

import torch

logger = logging.getLogger(__name__)


def get_optimal_device(verbose: bool = True) -> torch.device:
    """
    Return the best available torch.device with automatic CPU fallback.

    Checks for CUDA availability, selects cuda:0 if found, and logs the
    device name and free VRAM. Falls back to CPU with a warning if CUDA is
    unavailable or if torch is misconfigured.

    Parameters
    ----------
    verbose : bool
        When True, prints device name and VRAM info at INFO level.
        Set to False during unit tests or repeated calls.

    Returns
    -------
    torch.device
        'cuda:0' if a CUDA-capable GPU is available, otherwise 'cpu'.
    """
    if not torch.cuda.is_available():
        if verbose:
            logger.info("[gpu_utils] CUDA not available — using CPU.")
        return torch.device("cpu")

    device = torch.device("cuda:0")

    if verbose:
        props = torch.cuda.get_device_properties(0)
        # mem_get_info returns (free_bytes, total_bytes) for the current device.
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / 1024 ** 3
        total_gb = total_bytes / 1024 ** 3
        logger.info(
            "[gpu_utils] Using %s | VRAM: %.1f GB free / %.1f GB total | "
            "Compute capability: %d.%d",
            props.name,
            free_gb,
            total_gb,
            props.major,
            props.minor,
        )

    return device


def try_init_rmm_pool(
    initial_gb: float = 4.0,
    max_gb: float = None,
) -> bool:
    """
    Attempt to initialise an RMM (RAPIDS Memory Manager) pool allocator
    and integrate it with PyTorch's CUDA allocator.

    WHY RMM: for large-scale inference loops that allocate and free many
    tensors per batch, the default CUDA allocator can fragment the memory
    heap. RMM's pool allocator pre-reserves a contiguous block and serves
    sub-allocations from it, reducing fragmentation and allocation latency.

    The entire function is guarded so that any import error, CUDA error, or
    version mismatch silently returns False. The pipeline must not depend on
    RMM being present.

    Parameters
    ----------
    initial_gb : float
        Initial pool size in GB. Default 4 GB is conservative and safe on
        most 8+ GB cards. Increase for H100/A100 workloads.
    max_gb : float, optional
        Maximum pool size. If None, capped at 90% of total VRAM to leave
        headroom for CUDA runtime and PyTorch internal buffers.

    Returns
    -------
    bool
        True if RMM pool was initialised successfully, False otherwise.
    """
    try:
        import rmm
        from rmm.allocators.torch import rmm_torch_allocator

        if not torch.cuda.is_available():
            return False

        total_bytes = torch.cuda.get_device_properties(0).total_memory
        total_gb = total_bytes / 1024 ** 3

        # Cap max pool at 90% of total VRAM when not specified.
        if max_gb is None:
            max_gb = total_gb * 0.90

        initial_bytes = int(initial_gb * 1024 ** 3)
        max_bytes = int(max_gb * 1024 ** 3)

        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=initial_bytes,
            maximum_pool_size=max_bytes,
        )
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

        logger.info(
            "[gpu_utils] RMM pool initialised: %.1f GB initial / %.1f GB max.",
            initial_gb,
            max_gb,
        )
        return True

    except ImportError:
        # RMM not installed — this is expected on most systems.
        return False
    except Exception as exc:
        warnings.warn(
            f"[gpu_utils] RMM init failed ({exc}). "
            f"Continuing with default PyTorch allocator.",
            stacklevel=2,
        )
        return False


def clear_gpu_cache() -> None:
    """
    Release cached but un-allocated GPU memory back to the OS.
    Call between training epochs or before inference to reduce peak VRAM.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
