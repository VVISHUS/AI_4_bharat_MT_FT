"""Regression tests for the eval-during-training switch.

This setting was silently flipped back on twice while changing other things,
each time costing a multi-hour run: with the KV cache disabled (which both
IndicTrans2 checkpoints force), generation-based evaluation is quadratic in
output length and a single eval pass took ~50 minutes.

Reading the YAML is not enough -- what matters is what
`Seq2SeqTrainingArguments` actually ends up with after transformers' own
`__post_init__` normalisation. These tests construct the real object.

Run with:
    python -m pytest tests/test_training_args.py -v
    python tests/test_training_args.py            # standalone
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("USE_TF", "0")  # don't drag TensorFlow in
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# `datasets` is only needed for real data loading; stub it so these tests run
# in a bare environment. transformers probes __spec__, so the stub needs one.
if "datasets" not in sys.modules:
    _stub = types.ModuleType("datasets")
    _stub.__spec__ = importlib.machinery.ModuleSpec("datasets", None)
    _stub.Dataset = object
    _stub.load_dataset = lambda *args, **kwargs: None
    sys.modules["datasets"] = _stub

from src.config import load_config  # noqa: E402
from src.train import build_training_args  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "finetune_en_mr.yaml"


def _args(overrides=None, smoke=False):
    cfg = load_config(CONFIG, overrides or [])
    return cfg, build_training_args(cfg, Path("outputs/_test"), smoke=smoke)


def _strategy(args) -> str:
    value = getattr(args, "eval_strategy", None) or getattr(args, "evaluation_strategy", None)
    return str(value).upper()


def test_shipped_config_has_eval_disabled():
    """The config we actually ship must not evaluate during training."""
    cfg, args = _args()
    assert cfg["training"]["eval_during_training"] is False, (
        "configs/finetune_en_mr.yaml re-enabled eval during training. With the KV "
        "cache off this is extremely expensive -- see the config comment."
    )
    assert "NO" in _strategy(args)
    assert args.do_eval is False
    assert args.eval_steps is None


def test_disabling_eval_also_disables_generation_and_best_model():
    """No eval loop means nothing can rank checkpoints, so these must be off."""
    _, args = _args(["training.eval_during_training=false"])
    assert args.predict_with_generate is False
    assert args.load_best_model_at_end is False
    # Checkpoints must still be written, or the run produces nothing.
    assert "STEPS" in str(args.save_strategy).upper()


def test_enabling_eval_wires_the_right_best_metric():
    """With generation off there is no chrF++, so ranking must use eval_loss."""
    _, args = _args([
        "training.eval_during_training=true",
        "training.predict_with_generate=false",
    ])
    assert "STEPS" in _strategy(args)
    assert args.metric_for_best_model == "eval_loss"
    assert args.greater_is_better is False, "lower loss is better"


def test_enabling_generation_ranks_on_chrf():
    _, args = _args([
        "training.eval_during_training=true",
        "training.predict_with_generate=true",
    ])
    assert args.metric_for_best_model == "chrf++"
    assert args.greater_is_better is True


def test_smoke_mode_never_evaluates():
    """--smoke overfits 32 pairs; an eval loop there is pure waste."""
    _, args = _args(smoke=True)
    assert "NO" in _strategy(args)
    assert args.predict_with_generate is False


def test_precision_flags_are_mutually_exclusive():
    """fp16 and bf16 must never both be set, whatever the hardware."""
    _, args = _args()
    assert not (args.fp16 and args.bf16)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
