"""YAML config loading with dotted CLI overrides.

Kept deliberately tiny -- the alternative (hydra/omegaconf) is more dependency
surface than a one-day assignment justifies, and plain dicts are easier to dump
into the run directory for reproducibility.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


def _coerce(value: str) -> Any:
    """Turn a CLI string into the obvious Python type ('3' -> 3, 'true' -> True)."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply `a.b.c=value` strings onto a nested dict, in place-ish."""
    cfg = copy.deepcopy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"--set expects key=value, got {override!r}")
        dotted, raw = override.split("=", 1)
        node = cfg
        *parents, leaf = dotted.split(".")
        for part in parents:
            if part not in node:
                raise KeyError(f"unknown config section {part!r} in {dotted!r}")
            node = node[part]
        node[leaf] = _coerce(raw)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return apply_overrides(cfg, overrides or [])


def save_config(cfg: dict, output_dir: str | Path) -> Path:
    """Snapshot the *resolved* config next to the checkpoints.

    Without this, a run with CLI overrides is unreproducible three days later.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "resolved_config.json"
    target.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default="configs/finetune_en_mr.yaml",
        help="Path to the YAML config.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted overrides, e.g. --set training.learning_rate=3e-5",
    )
    return parser
