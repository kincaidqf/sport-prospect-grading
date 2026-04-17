import os
import torch


def get_device() -> torch.device:
    """Return the best available compute device, respecting DEVICE env override."""
    override = os.environ.get("DEVICE", "").lower()

    if override == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda requested but no CUDA device found")
        return torch.device("cuda")

    if override == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("DEVICE=mps requested but MPS is not available")
        return torch.device("mps")

    if override == "cpu":
        return torch.device("cpu")

    # Auto-detect: CUDA > MPS > CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_device_info(device: torch.device) -> None:
    print(f"[device] using: {device}")
    if device.type == "cuda":
        idx = device.index or 0
        print(f"[device]   {torch.cuda.get_device_name(idx)}")
        total = torch.cuda.get_device_properties(idx).total_memory / 1e9
        print(f"[device]   VRAM: {total:.1f} GB")
    elif device.type == "mps":
        print("[device]   Apple Silicon MPS backend")
    else:
        print("[device]   CPU-only training")
