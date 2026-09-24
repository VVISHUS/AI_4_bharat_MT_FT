"""Evaluate base vs fine-tuned on a held-out benchmark.

Usage
-----
    python -m src.evaluate --checkpoint outputs/rotary-it2-en-mr/final
    python -m src.evaluate --checkpoint ... --base-only    # baseline first

Design notes
------------
* The benchmark is IN22-Gen (AI4Bharat's own), not the Samanantar validation
  split. Evaluating on held-out rows of a mined corpus measures how well we fit
  the miner's noise, which is not the question being asked.
* chrF++ is the headline metric. BLEU on a morphologically rich target with a
  single reference is noisy and rewards the wrong things; we report it because
  it is expected, not because it is informative here.
* Base and fine-tuned are decoded with identical generation settings so the
  comparison is about weights, not sampling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import sacrebleu
import torch
from datasets import load_dataset

from src.config import base_arg_parser, load_config
from src.data import ProcessorAdapter, decode_targets, detect_label_strategy, encode_sources
from src.modeling import load_model, load_tokenizer, pick_dtype

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
LOGGER = logging.getLogger("evaluate")


def load_benchmark(eval_cfg: dict, src_lang: str, tgt_lang: str):
    """Load the first benchmark candidate that works.

    Hub packaging for these benchmarks drifts. IN22-Gen has been repackaged as
    parquet, so the per-pair config names on its card ("eng_Latn-mar_Deva") may
    no longer exist and its split is "gen" rather than "test". FLORES is gated
    and so cannot serve as a silent fallback. Rather than hardcode one shape and
    discover the mistake at evaluation time, try the plausible ones and say
    which worked.
    """
    candidates = eval_cfg.get("candidates")
    if not candidates:  # legacy config shape
        candidates = [
            {"dataset": eval_cfg["dataset"], "config": eval_cfg.get("dataset_config"),
             "split": eval_cfg.get("split", "test")},
            {"dataset": eval_cfg["fallback_dataset"], "config": eval_cfg.get("fallback_config"),
             "split": eval_cfg.get("fallback_split", "devtest")},
        ]

    errors: list[str] = []
    for candidate in candidates:
        name = candidate["dataset"]
        config = candidate.get("config")
        split = candidate.get("split")
        label = f"{name}/{config or '<default>'}[{split}]"
        try:
            dataset = load_dataset(name, config, split=split)
            LOGGER.info("Benchmark loaded: %s (%d rows)", label, len(dataset))
            return dataset, label
        except Exception as exc:  # noqa: BLE001 - any failure means "try the next"
            short = str(exc).split("\n")[0][:160]
            LOGGER.warning("  x %s -> %s", label, short)
            errors.append(f"{label}: {short}")

    raise RuntimeError(
        "No benchmark could be loaded. Tried:\n  "
        + "\n  ".join(errors)
        + "\n\nIN22-Gen and FLORES are both GATED. Accept the terms while logged "
        "in and set HF_TOKEN:\n"
        "  https://huggingface.co/datasets/ai4bharat/IN22-Gen\n"
        "Note this is separate from accepting the model's terms.\n"
        "Alternatively run with --in-domain to score the held-out Samanantar "
        "split instead, which needs no extra access."
    )


def load_in_domain_split(cfg: dict):
    """Rebuild the held-out Samanantar validation split used during training.

    Deterministic: the same filters over the same stream, shuffled with the same
    seed, yield the same rows the training run held out.

    Scoring this *alongside* the out-of-domain benchmark is the point. The gap
    between them is the evidence: a fine-tune that improves in-domain while
    flat or worse out-of-domain has learned the mined corpus's quirks rather
    than better Marathi.
    """
    from datasets import Dataset

    from src.data import load_samanantar_pairs

    data_cfg = cfg["data"]
    pairs, _ = load_samanantar_pairs(data_cfg)
    dataset = Dataset.from_list(pairs).shuffle(seed=cfg["seed"])
    n_valid = min(data_cfg["valid_samples"], max(1, len(dataset) // 10))
    valid = dataset.select(range(n_valid))
    LOGGER.info("In-domain split: %d held-out Samanantar pairs", len(valid))
    return valid, "samanantar-mr (held-out, in-domain)"


def resolve_columns(dataset, src_lang: str, tgt_lang: str) -> tuple[str, str]:
    """Find the source/target text columns by language tag.

    IN22-Gen and FLORES both name columns like `sentence_eng_Latn`, but the
    prefix has varied, so we match on the tag rather than the full name.
    """
    columns = dataset.column_names
    src_col = next((c for c in columns if src_lang in c), None)
    tgt_col = next((c for c in columns if tgt_lang in c), None)
    if src_col is None or tgt_col is None:
        raise RuntimeError(
            f"Could not find columns for {src_lang}/{tgt_lang} in {columns}. "
            "Pass them explicitly via --src-column/--tgt-column."
        )
    return src_col, tgt_col


@torch.inference_mode()
def translate(
    model,
    tokenizer,
    processor: ProcessorAdapter,
    sentences: list[str],
    strategy: str,
    src_lang: str,
    tgt_lang: str,
    batch_size: int = 16,
    num_beams: int = 5,
    max_length: int = 256,
) -> list[str]:
    device = next(model.parameters()).device
    outputs: list[str] = []

    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start : start + batch_size]
        prepared = processor.source(chunk, src_lang, tgt_lang)
        encoded = encode_sources(tokenizer, prepared, strategy, max_length)
        batch = tokenizer.pad(encoded, return_tensors="pt").to(device)

        generated = model.generate(
            **batch,
            num_beams=num_beams,
            max_length=max_length,
            # Greedy-ish length control: MT wants no repetition penalty games,
            # just beams. Keep this identical between base and tuned.
            early_stopping=True,
            # Required for this checkpoint -- see modeling.disable_kv_cache().
            use_cache=False,
        )
        decoded = decode_targets(tokenizer, generated, strategy)
        # postprocess_batch restores entities/numerals the processor masked.
        outputs.extend(processor.postprocess(decoded, lang=tgt_lang))
        LOGGER.info("translated %d/%d", min(start + batch_size, len(sentences)), len(sentences))

    return outputs


def score(hypotheses: list[str], references: list[str]) -> dict:
    refs = [[r.strip() for r in references]]
    hyps = [h.strip() for h in hypotheses]
    chrf = sacrebleu.corpus_chrf(hyps, refs, word_order=2)
    bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize="none")
    return {"chrf++": round(chrf.score, 3), "bleu": round(bleu.score, 3)}


def qualitative_table(sources, references, base_out, tuned_out, limit: int) -> str:
    lines = [
        "| # | English source | Reference (mr) | Base | Fine-tuned |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for i in range(min(limit, len(sources))):
        cells = [
            str(i + 1),
            sources[i].replace("|", "\\|"),
            references[i].replace("|", "\\|"),
            (base_out[i] if base_out else "-").replace("|", "\\|"),
            (tuned_out[i] if tuned_out else "-").replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = base_arg_parser("Evaluate base vs fine-tuned IndicTrans2 on en->mr.")
    parser.add_argument("--checkpoint", default=None, help="Path to the fine-tuned model/adapter.")
    parser.add_argument("--base-only", action="store_true", help="Score the base model alone.")
    parser.add_argument("--output", default=None, help="Where to write results (default: alongside checkpoint).")
    parser.add_argument(
        "--in-domain",
        action="store_true",
        help="Score the held-out Samanantar split instead of the benchmark. "
             "Needs no gated access, and the in/out-of-domain gap is informative.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    ecfg = cfg["evaluation"]
    src_lang = cfg["data"]["src_lang"]
    tgt_lang = cfg["data"]["tgt_lang"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(cfg["model"].get("dtype", "auto"))

    tokenizer = load_tokenizer(cfg["model"])
    processor = ProcessorAdapter(inference=True)
    strategy = detect_label_strategy(tokenizer)
    LOGGER.info("device=%s dtype=%s label_strategy=%s", device, dtype, strategy)

    if args.in_domain:
        dataset, bench_name = load_in_domain_split(cfg)
        src_col, tgt_col = "src", "tgt"
    else:
        dataset, bench_name = load_benchmark(ecfg, src_lang, tgt_lang)
        src_col, tgt_col = resolve_columns(dataset, src_lang, tgt_lang)

    n = min(ecfg["max_eval_samples"], len(dataset))
    dataset = dataset.select(range(n))
    sources = [s.strip() for s in dataset[src_col]]
    references = [s.strip() for s in dataset[tgt_col]]

    gen_kwargs = dict(
        strategy=strategy,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        num_beams=ecfg["num_beams"],
        max_length=ecfg["max_length"],
    )

    results: dict = {"benchmark": bench_name, "n_samples": n}

    LOGGER.info("--- baseline ---")
    base_model = load_model(cfg["model"], dtype=dtype).to(device).eval()
    base_out = translate(base_model, tokenizer, processor, sources, **gen_kwargs)
    results["base"] = score(base_out, references)
    LOGGER.info("base: %s", results["base"])

    tuned_out: list[str] = []
    if not args.base_only:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --base-only is set.")
        LOGGER.info("--- fine-tuned ---")
        del base_model
        torch.cuda.empty_cache()

        checkpoint = Path(args.checkpoint)
        if (checkpoint / "adapter_config.json").exists():
            # LoRA run: reload the base, then apply the adapter on top.
            from peft import PeftModel

            tuned = load_model(cfg["model"], dtype=dtype)
            tuned = PeftModel.from_pretrained(tuned, str(checkpoint))
            tuned = tuned.merge_and_unload()  # fold adapters in for faster decode
            LOGGER.info("Loaded LoRA adapter from %s", checkpoint)
        else:
            tuned = load_model({**cfg["model"], "name": str(checkpoint)}, dtype=dtype)
            LOGGER.info("Loaded full checkpoint from %s", checkpoint)

        tuned = tuned.to(device).eval()
        tuned_out = translate(tuned, tokenizer, processor, sources, **gen_kwargs)
        results["finetuned"] = score(tuned_out, references)
        LOGGER.info("finetuned: %s", results["finetuned"])
        results["delta"] = {
            k: round(results["finetuned"][k] - results["base"][k], 3)
            for k in results["base"]
        }

    # Results land next to the checkpoint by default, so a run directory is
    # self-describing: config + filter report + train metrics + eval results.
    out_dir = Path(args.output) if args.output else Path(args.checkpoint or "outputs").parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Distinct filenames so an in-domain run does not clobber the benchmark run.
    suffix = "_in_domain" if args.in_domain else ""
    (out_dir / f"eval_results{suffix}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / f"qualitative_samples{suffix}.md").write_text(
        qualitative_table(
            sources, references, base_out, tuned_out, ecfg["num_qualitative_samples"]
        ),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    LOGGER.info("Wrote results to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
