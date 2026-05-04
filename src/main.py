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
        choices=["regression", "classification", "text", "text_shallow", "multimodal"],
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
    if model_type not in {"regression", "classification", "multimodal"}:
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
    elif model_type == "multimodal":
        import src.models.multimodal as mm
        mm.run(
            cfg=cfg,
            run_name=args.run_name,
            tracking_uri=args.tracking_uri,
        )
    elif model_type == "text":
        import src.models.text_model as tm
        text_cfg = cfg["model"].get("text", {})
        train_cfg = cfg.get("training") or {}
        tm.train_and_evaluate_text_model(
            pretrained=text_cfg.get("pretrained", "distilbert-base-uncased"),
            output_dim=int(text_cfg.get("output_dim", 64)),
            hidden_dim=int(text_cfg.get("hidden_dim", 32)),
            dropout=float(text_cfg.get("dropout", 0.2)),
            freeze_base=bool(text_cfg.get("freeze_base", True)),
            max_length=int(text_cfg.get("max_length", 128)),
            huber_beta=float(text_cfg.get("huber_beta", 1.0)),
            task=text_cfg.get("task"),
            classification_target_col=text_cfg.get("classification_target_col"),
            num_classes=int(text_cfg.get("num_classes", 4)),
            batch_size=int(train_cfg.get("batch_size", 32)),
            epochs=int(train_cfg.get("epochs", 3)),
            lr=float(train_cfg.get("lr", 1e-3)),
            cfg=cfg,
            run_name=args.run_name,
            tracking_uri=args.tracking_uri,
            regression_target_col=text_cfg.get("regression_target_col"),
            tier_proba_csv_path=text_cfg.get("tier_proba_csv_path"),
        )
    elif model_type == "text_shallow":
        import src.models.simple_text_model as stm
        text_cfg = cfg["model"].get("text", {})
        stm.train_and_evaluate_shallow_text_model(
            cfg=cfg,
            run_name=args.run_name,
            tracking_uri=args.tracking_uri,
            tier_proba_csv_path=text_cfg.get("tier_proba_csv_path"),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    main()
