"""Resolve multimodal text stack config (parallel to base_models.classification / regression)."""
from __future__ import annotations


def normalize_text_tower_entry(raw: dict) -> dict:
    meta_key = str(raw.get("meta_key") or "").strip()
    if not meta_key:
        raise ValueError("multimodal.text_towers entries require a non-empty meta_key")
    typ = str(raw.get("type", "transformer")).strip().lower()
    shallow = raw.get("shallow")
    if shallow is not None and not isinstance(shallow, dict):
        raise TypeError(f"text tower {meta_key!r}: shallow must be a mapping or null")
    return {"meta_key": meta_key, "type": typ, "shallow": dict(shallow or {})}


def resolve_text_stack(mm_cfg: dict) -> tuple[list[str], list[dict]]:
    """Return ``(text_model_keys, tower_specs)`` for stacking.

    **Precedence**

    1. ``multimodal.base_models.text`` — list of meta keys, same pattern as
       ``base_models.classification`` / ``regression``. Pair with
       ``multimodal.text_backends`` (key → ``transformer`` | ``shallow_lexicon``)
       and optional ``text_shallow_overrides`` for per-key shallow YAML.
    2. Legacy ``multimodal.text_towers`` — explicit list of tower dicts (used when
       ``base_models.text`` is not set).
    3. Single ``text_meta_key`` → one transformer tower (original default).

    Tower spec dicts: ``{"meta_key", "type", "shallow"}`` (``shallow`` merges onto
    ``model.text_shallow`` for shallow types).
    """
    base_models = mm_cfg.get("base_models") or {}
    text_from_base = base_models.get("text")

    if text_from_base is not None:
        if not isinstance(text_from_base, list):
            raise TypeError("multimodal.base_models.text must be a list of strings")
        text_models = [str(k).strip() for k in text_from_base if str(k).strip()]
        if not text_models:
            raise ValueError("multimodal.base_models.text is empty")
        backends = mm_cfg.get("text_backends") or {}
        overrides = mm_cfg.get("text_shallow_overrides") or {}
        tower_specs: list[dict] = []
        for key in text_models:
            typ = str(backends.get(key, "transformer")).strip().lower()
            if typ in ("", "deep", "bert"):
                typ = "transformer"
            raw_ov = overrides.get(key)
            shallow = dict(raw_ov) if isinstance(raw_ov, dict) else {}
            tower_specs.append({"meta_key": key, "type": typ, "shallow": shallow})
        return text_models, tower_specs

    towers_raw = mm_cfg.get("text_towers")
    if towers_raw:
        if not isinstance(towers_raw, list):
            raise TypeError("multimodal.text_towers must be a list")
        specs = [normalize_text_tower_entry(t) for t in towers_raw]
        return [s["meta_key"] for s in specs], specs

    key = str(mm_cfg.get("text_meta_key", "scouting")).strip() or "scouting"
    return [key], [{"meta_key": key, "type": "transformer", "shallow": {}}]


def parse_text_towers(mm_cfg: dict) -> list[dict]:
    """Return tower specs only (legacy helper; prefer :func:`resolve_text_stack`)."""
    _, specs = resolve_text_stack(mm_cfg)
    return specs
