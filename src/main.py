"""Entry point — loads config, resolves device, and dispatches to the selected model pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.utils.device import get_device, log_device_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NBA Draft Prospect ML")
    p.add_argument("--config", default="src/config/config.yaml")
    p.add_argument(
        "--model",
        choices=["regression", "text", "multimodal"],
        default=None,
        help="Override model.type from config",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.model:
        cfg["model"]["type"] = args.model
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir

    device = get_device()
    log_device_info(device)

    model_type = cfg["model"]["type"]
    print(f"[config] model: {model_type}")
    print(f"[config] epochs: {cfg['training']['epochs']}")
    print(f"[config] output: {cfg['output']['dir']}")

    if model_type == "regression":
        # TODO: from src.training.run_regression import run; run(cfg, device)
        raise NotImplementedError("regression pipeline not yet implemented")
    elif model_type == "text":
        # TODO: from src.training.run_text import run; run(cfg, device)
        raise NotImplementedError("text pipeline not yet implemented")
    elif model_type == "multimodal":
        # TODO: from src.training.run_multimodal import run; run(cfg, device)
        raise NotImplementedError("multimodal pipeline not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    main()
