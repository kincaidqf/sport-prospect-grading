"""Shared MLflow configuration and run-management helpers."""
from __future__ import annotations

import getpass
import os
import socket
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_TRACKING_DIR = PROJECT_ROOT / "mlruns"
DEFAULT_LOCAL_ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"
DEFAULT_LOCAL_PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


@dataclass
class MLflowContext:
    tracking_uri: str
    experiment_name: str
    parent_run_name: str
    artifact_location: str | None
    plot_dir: str
    tags: dict[str, str]


def _looks_like_uri(value: str) -> bool:
    parsed = urlparse(value)
    return bool(parsed.scheme) and parsed.scheme != "file"


def _to_file_uri(path_value: str | os.PathLike[str]) -> str:
    return Path(path_value).expanduser().resolve().as_uri()


def _resolve_tracking_uri(cfg: dict[str, Any] | None = None, tracking_uri: str | None = None) -> str:
    logging_cfg = ((cfg or {}).get("logging") or {}).get("mlflow", {})
    raw = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or logging_cfg.get("tracking_uri")
    if raw:
        if raw.startswith("file://") or raw.startswith("sqlite:") or _looks_like_uri(raw):
            return raw
        return _to_file_uri(raw)
    return _to_file_uri(DEFAULT_LOCAL_TRACKING_DIR)


def _resolve_artifact_location(cfg: dict[str, Any] | None = None, artifact_location: str | None = None) -> str | None:
    logging_cfg = ((cfg or {}).get("logging") or {}).get("mlflow", {})
    raw = artifact_location or os.getenv("MLFLOW_ARTIFACT_LOCATION") or logging_cfg.get("artifact_location")
    if raw:
        if raw.startswith("file://") or _looks_like_uri(raw):
            return raw
        return _to_file_uri(raw)
    return None


def _resolve_plots_base_dir(cfg: dict[str, Any] | None = None) -> Path:
    output_cfg = (cfg or {}).get("output") or {}
    raw = os.getenv("MODEL_PLOTS_DIR") or output_cfg.get("plots_dir")
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_LOCAL_PLOTS_DIR


def _get_git_value(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _default_run_name(model_type: str, target_name: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = model_type if not target_name else f"{model_type}-{target_name}"
    user = getpass.getuser().replace(" ", "-")
    short_sha = (_get_git_value(["git", "rev-parse", "--short", "HEAD"]) or "nogit").lower()
    return f"{slug}-{timestamp}-{user}-{short_sha}"


def _default_experiment_name(cfg: dict[str, Any] | None, model_type: str, fallback: str | None) -> str:
    logging_cfg = ((cfg or {}).get("logging") or {}).get("mlflow", {})
    configured = logging_cfg.get("experiment_name")
    if configured:
        return configured
    prefix = logging_cfg.get("experiment_prefix") or (cfg or {}).get("logging", {}).get("project") or "nba-draft-ml"
    return fallback or f"{prefix}-{model_type}"


def build_mlflow_context(
    *,
    cfg: dict[str, Any] | None = None,
    model_type: str,
    target_name: str | None = None,
    fallback_experiment_name: str | None = None,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
    run_name: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> MLflowContext:
    """Load env/config, set the tracking URI, and prepare shared run metadata."""
    load_dotenv(PROJECT_ROOT / ".env")

    resolved_tracking_uri = _resolve_tracking_uri(cfg=cfg, tracking_uri=tracking_uri)
    resolved_artifact_location = _resolve_artifact_location(cfg=cfg, artifact_location=artifact_location)
    experiment_name = _default_experiment_name(cfg, model_type, fallback_experiment_name)
    parent_run_name = run_name or os.getenv("MLFLOW_RUN_NAME") or _default_run_name(model_type, target_name)
    plot_dir = _resolve_plots_base_dir(cfg) / parent_run_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(resolved_tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        if resolved_artifact_location:
            mlflow.create_experiment(experiment_name, artifact_location=resolved_artifact_location)
        else:
            mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    branch = _get_git_value(["git", "branch", "--show-current"])
    commit = _get_git_value(["git", "rev-parse", "HEAD"])

    tags = {
        "project_root": str(PROJECT_ROOT),
        "model_type": model_type,
        "target_name": target_name or "",
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "git_branch": branch or "",
        "git_commit": commit or "",
        "tracking_uri": resolved_tracking_uri,
    }
    if extra_tags:
        tags.update({key: str(value) for key, value in extra_tags.items()})

    print(f"[mlflow] tracking_uri: {resolved_tracking_uri}")
    print(f"[mlflow] experiment: {experiment_name}")
    print(f"[mlflow] run_name: {parent_run_name}")
    print(f"[mlflow] plot_dir: {plot_dir}")

    return MLflowContext(
        tracking_uri=resolved_tracking_uri,
        experiment_name=experiment_name,
        parent_run_name=parent_run_name,
        artifact_location=resolved_artifact_location,
        plot_dir=str(plot_dir),
        tags=tags,
    )


@contextmanager
def managed_run(
    ctx: MLflowContext,
    *,
    run_name: str | None = None,
    nested: bool = False,
    tags: dict[str, str] | None = None,
):
    """Start an MLflow run with shared tags and naming conventions."""
    merged_tags = dict(ctx.tags)
    if tags:
        merged_tags.update({key: str(value) for key, value in tags.items()})
    with mlflow.start_run(run_name=run_name or ctx.parent_run_name, nested=nested, tags=merged_tags) as run:
        yield run


def log_config_dict(cfg: dict[str, Any]) -> None:
    """Persist the full resolved config as an MLflow artifact."""
    mlflow.log_dict(cfg, "config/config.json")


def log_common_params(params: dict[str, Any]) -> None:
    """Log params after coercing lists/bools into MLflow-friendly strings."""
    cleaned = {}
    for key, value in params.items():
        if isinstance(value, (list, tuple, dict)):
            cleaned[key] = str(value)
        else:
            cleaned[key] = value
    mlflow.log_params(cleaned)


def log_epoch_metrics(metrics: dict[str, float], epoch: int) -> None:
    """Log epoch-indexed metrics."""
    for key, value in metrics.items():
        mlflow.log_metric(key, float(value), step=epoch)


def log_reproducibility_metadata(device: str = "cpu") -> None:
    """Log Python version, key library versions, and compute device to the active run."""
    import sys

    params: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "device": device,
    }
    for lib in ("sklearn", "xgboost", "mlflow", "torch", "transformers"):
        try:
            mod = __import__(lib)
            params[f"{lib}_version"] = mod.__version__
        except ImportError:
            pass
    log_common_params(params)


def log_data_summary(
    df: Any,
    target_col: str,
    task: str,
    test_size: float,
    cv_folds: int,
    random_seed: int,
) -> None:
    """Log dataset size, split config, and target distribution to the active run."""
    n_total = len(df)
    n_test = round(n_total * test_size)
    n_train = n_total - n_test

    params: dict[str, Any] = {
        "n_total": n_total,
        "n_train": n_train,
        "n_test": n_test,
        "test_size": test_size,
        "cv_folds": cv_folds,
        "random_seed": random_seed,
        "split_type": "random",
    }

    col = df[target_col].dropna()
    if task == "regression":
        params.update(
            {
                "target_mean": round(float(col.mean()), 4),
                "target_std": round(float(col.std()), 4),
                "target_min": round(float(col.min()), 4),
                "target_max": round(float(col.max()), 4),
            }
        )
    elif task == "classification":
        n_pos = int(col.sum())
        params.update(
            {
                "n_positive": n_pos,
                "n_negative": n_total - n_pos,
                "class_balance": round(float(col.mean()), 4),
            }
        )

    log_common_params(params)


def log_candidate_summary(results: dict[str, Any], task: str) -> None:
    """Log best-model params and a JSON candidate-summary artifact to the active run."""
    import json
    import os
    import tempfile

    selection_metric = "r2" if task == "regression" else "auc"

    summary: dict[str, Any] = {}
    for name, res in results.items():
        if task == "regression":
            entry: dict[str, Any] = {
                "test_r2": res["r2"],
                "test_rmse": res["rmse"],
                "test_mae": res["mae"],
            }
        else:
            entry = {
                "test_accuracy": res["accuracy"],
                "test_roc_auc": res["auc"],
            }
        if res.get("best_cv_score") is not None:
            entry["best_cv_score"] = res["best_cv_score"]
        summary[name] = entry

    best_name = max(results, key=lambda n: results[n].get(selection_metric, float("-inf")))
    log_common_params(
        {
            "best_model": best_name,
            f"best_{selection_metric}": results[best_name].get(selection_metric),
        }
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="candidate_summary_"
    ) as f:
        json.dump(summary, f, indent=2)
        tmp_path = f.name

    try:
        mlflow.log_artifact(tmp_path, artifact_path="summary")
    finally:
        os.unlink(tmp_path)
