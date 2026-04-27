"""Entry point — loads config, resolves device, and dispatches to the selected model pipeline."""
from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

import yaml

from src.utils.device import get_device, log_device_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NBA Draft Prospect ML")
    p.add_argument("--config", default="src/config/config.yaml")
    p.add_argument(
        "--model",
        choices=["regression", "classification", "text", "multimodal"],
        default=None,
        help="Override model.type from config",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--tracking-uri", default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _ensure_macos_openmp_for_xgboost(model_type: str) -> None:
    """Restart with a known OpenMP path when macOS XGBoost cannot find libomp."""
    if model_type not in {"regression", "classification"}:
        return
    if platform.system() != "Darwin":
        return
    if os.environ.get("XGBOOST_OPENMP_REEXECED") == "1":
        return

    fallback_var = "DYLD_FALLBACK_LIBRARY_PATH"
    existing_paths = os.environ.get(fallback_var, "").split(os.pathsep)
    candidates = [
        Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
        Path("/usr/local/opt/libomp/lib/libomp.dylib"),
        Path("/opt/miniconda3/lib/libomp.dylib"),
        Path.home() / "miniconda3/lib/libomp.dylib",
        Path.home() / "anaconda3/lib/libomp.dylib",
    ]
    for libomp_path in candidates:
        if not libomp_path.exists():
            continue
        libomp_dir = str(libomp_path.parent)
        if libomp_dir in existing_paths:
            return

        env = os.environ.copy()
        env["XGBOOST_OPENMP_REEXECED"] = "1"
        env["PYTHONUNBUFFERED"] = "1"  # prevent SIGSEGV during stdout flush at exit
        env["OMP_NUM_THREADS"] = "1"   # prevent multi-runtime OpenMP conflict (SAGA + XGBoost)
        env[fallback_var] = os.pathsep.join(
            [libomp_dir, *[path for path in existing_paths if path]]
        )
        print(f"[xgboost] restarting with {fallback_var}={libomp_dir}", flush=True)
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.model:
        cfg["model"]["type"] = args.model
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir

    model_type = cfg["model"]["type"]
    _ensure_macos_openmp_for_xgboost(model_type)

    device = get_device()
    log_device_info(device)

    print(f"[config] model: {model_type}")
    print(f"[config] epochs: {cfg['training']['epochs']}")
    print(f"[config] output: {cfg['output']['dir']}")

    if model_type == "regression":
        import src.models.regression_model as rm
        regression_cfg = cfg["model"].get("regression", {})
        rm.run(
            target_mode=regression_cfg.get("target_mode", "composite_score"),
            use_draft_pick=regression_cfg.get("use_draft_pick", False),
            cfg=cfg,
            run_name=args.run_name,
            tracking_uri=args.tracking_uri,
        )
    elif model_type == "classification":
        import src.models.classification_model as cm
        classification_cfg = cfg["model"].get("classification", {})
        cm.run(
            target_mode=classification_cfg.get("target_mode", "prospect_tier"),
            use_draft_pick=classification_cfg.get("use_draft_pick", False),
            cfg=cfg,
            run_name=args.run_name,
            tracking_uri=args.tracking_uri,
        )
    elif model_type == "text":
        import src.models.text_model as tm
        text_cfg = cfg["model"].get("text", {})
        tm.train_and_evaluate_text_model(
            pretrained=text_cfg.get("pretrained", "distilbert-base-uncased"),
            output_dim=text_cfg.get("output_dim", 128),
            freeze_base=text_cfg.get("freeze_base", False),
            max_length=text_cfg.get("max_length", 512),
            batch_size=cfg["training"].get("batch_size", 16),
            epochs=cfg["training"].get("epochs", 3),
            lr=cfg["training"].get("lr", 2e-5),
            cfg=cfg,
            run_name=args.run_name,
            tracking_uri=args.tracking_uri,
        )
    elif model_type == "multimodal":
        # TODO: from src.training.run_multimodal import run; run(cfg, device)
        raise NotImplementedError("multimodal pipeline not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    main()
