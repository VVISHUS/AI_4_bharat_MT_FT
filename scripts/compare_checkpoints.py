"""Score every saved checkpoint on the held-out benchmark and rank them.

Why this exists
---------------
With evaluation during training disabled, the Trainer cannot rank checkpoints,
so `load_best_model_at_end` is off and the run simply keeps the last few.

The tempting substitute is to select the checkpoint with the lowest *training*
loss. That is a bad criterion: training loss measures fit to the batches just
trained on, so selecting on it biases toward the most overfit checkpoint. It is
also noisy -- ours jitters by +/-0.05 between 50-step windows.

The right criterion is the metric we actually care about, on data the model has
never seen: chrF++ on IN22-Gen, which is human-curated and out of domain. At
~95s per checkpoint that is cheap enough to just measure.

Usage
-----
    python scripts/compare_checkpoints.py
    python scripts/compare_checkpoints.py --run-dir outputs/rotary-it2-en-mr
    python scripts/compare_checkpoints.py --max-eval-samples 200   # quicker
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import ProcessorAdapter, detect_label_strategy  # noqa: E402
from src.evaluate import load_benchmark, resolve_columns, score, translate  # noqa: E402
from src.modeling import load_model, load_tokenizer, pick_dtype  # noqa: E402


def find_checkpoints(run_dir: Path) -> list[tuple[str, Path]]:
    """Return [(label, path)] for every checkpoint in the run directory, in order."""
    found: list[tuple[int, str, Path]] = []

    for path in sorted(run_dir.glob("checkpoint-*")):
        match = re.search(r"checkpoint-(\d+)$", path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path.name, path))

    final = run_dir / "final"
    if final.is_dir():
        # Sorts last: it is the end-of-training state.
        found.append((10**9, "final", final))

    found.sort()
    return [(label, path) for _, label, path in found]


def load_checkpoint(model_cfg: dict, checkpoint: Path, dtype):
    """Load a checkpoint, applying the LoRA adapter if that is what it is."""
    if (checkpoint / "adapter_config.json").exists():
        from peft import PeftModel

        model = load_model(model_cfg, dtype=dtype)
        model = PeftModel.from_pretrained(model, str(checkpoint))
        return model.merge_and_unload()
    return load_model({**model_cfg, "name": str(checkpoint)}, dtype=dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finetune_en_mr.yaml")
    parser.add_argument("--run-dir", default=None, help="Defaults to training.output_dir.")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--include-base", action="store_true", default=True,
                        help="Also score the untuned model, for reference.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ecfg = cfg["evaluation"]
    run_dir = Path(args.run_dir or cfg["training"]["output_dir"])
    if not run_dir.is_dir():
        raise SystemExit(f"No such run directory: {run_dir}")

    checkpoints = find_checkpoints(run_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {run_dir}")
    print(f"Found {len(checkpoints)} checkpoint(s): {[c[0] for c in checkpoints]}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(cfg["model"].get("dtype", "auto"))
    tokenizer = load_tokenizer(cfg["model"])
    processor = ProcessorAdapter(inference=True)
    strategy = detect_label_strategy(tokenizer)

    src_lang, tgt_lang = cfg["data"]["src_lang"], cfg["data"]["tgt_lang"]
    dataset, bench = load_benchmark(ecfg, src_lang, tgt_lang)
    src_col, tgt_col = resolve_columns(dataset, src_lang, tgt_lang)
    n = min(args.max_eval_samples or ecfg["max_eval_samples"], len(dataset))
    dataset = dataset.select(range(n))
    sources = [s.strip() for s in dataset[src_col]]
    references = [s.strip() for s in dataset[tgt_col]]
    print(f"Benchmark: {bench}, {n} samples, beams={ecfg['num_beams']}\n")

    gen_kwargs = dict(
        strategy=strategy, src_lang=src_lang, tgt_lang=tgt_lang,
        num_beams=ecfg["num_beams"], max_length=ecfg["max_length"],
    )

    results: list[dict] = []
    targets: list[tuple[str, Path | None]] = []
    if args.include_base:
        targets.append(("base (untuned)", None))
    targets.extend(checkpoints)

    for label, path in targets:
        print(f"--- {label} ---")
        model = (
            load_model(cfg["model"], dtype=dtype) if path is None
            else load_checkpoint(cfg["model"], path, dtype)
        )
        model = model.to(device).eval()
        hypotheses = translate(model, tokenizer, processor, sources, **gen_kwargs)
        row = {"checkpoint": label, **score(hypotheses, references)}
        results.append(row)
        print(f"  {row}\n")

        # Free VRAM before the next checkpoint; these are 218M-param models.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tuned = [r for r in results if r["checkpoint"] != "base (untuned)"]
    best = max(tuned, key=lambda r: r["chrf++"]) if tuned else None

    print("=" * 56)
    print(f"{'checkpoint':<20}{'chrF++':>10}{'BLEU':>10}")
    print("-" * 56)
    for row in results:
        marker = "  <- best" if best and row is best else ""
        print(f"{row['checkpoint']:<20}{row['chrf++']:>10.2f}{row['bleu']:>10.2f}{marker}")
    print("=" * 56)
    if best:
        print(f"\nSelected by chrF++ on held-out {bench}, NOT by training loss.")
        print(f"Best: {best['checkpoint']}")

    out = run_dir / "checkpoint_comparison.json"
    out.write_text(
        json.dumps({"benchmark": bench, "n_samples": n, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
