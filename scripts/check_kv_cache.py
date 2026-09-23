"""Does this checkpoint's remote code work with the modern transformers Cache API?

IndicTrans2 checkpoints ship their architecture as remote code on the Hub. Some
of those files were written against the pre-4.43 transformers API, where
`generate()` passed `past_key_values=None` on the first decoding step. Modern
transformers passes an empty `EncoderDecoderCache` instead, and the older code
then does:

    past_key_values[0][0].shape[2] if past_key_values is not None else 0

which takes the wrong branch and dies on `NoneType.shape`.

Without the KV cache, decoding recomputes the whole prefix at every step --
quadratic instead of linear in output length, roughly 60x more work for a
128-token translation. That is survivable for one final evaluation and fatal for
anything that generates repeatedly.

This script answers, for any checkpoint, in about a minute:

    python scripts/check_kv_cache.py                                  # config default
    python scripts/check_kv_cache.py --model ai4bharat/indictrans2-en-indic-dist-200M

It reports whether generation works with the cache ON, and how much slower OFF
is, so the choice of checkpoint is made on measurements rather than assumption.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from pathlib import Path

# Running as `python scripts/x.py` puts scripts/ on sys.path, not the repo root,
# so `import src...` fails. Add the repo root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SENTENCES = [
    "The committee published its final report on Tuesday morning.",
    "She has been working on this problem for several years.",
    "The new policy will take effect from the first of next month.",
    "Farmers in the region reported a significant drop in yield.",
]


def rule(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def timed_generate(model, tokenizer, processor, sentences, use_cache, num_beams, max_length):
    """Translate once and return (seconds, outputs) or (None, error string)."""
    prepared = processor.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="mar_Deva")
    encoded = tokenizer(prepared, return_tensors="pt", padding=True, truncation=True)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=True,
                use_cache=use_cache,
            )
    except Exception as exc:  # noqa: BLE001 - reporting the failure IS the result
        return None, f"{type(exc).__name__}: {exc}"

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return elapsed, processor.postprocess_batch(decoded, lang="mar_Deva")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Checkpoint to test.")
    parser.add_argument("--config", default="configs/finetune_en_mr.yaml")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from IndicTransToolkit.processor import IndicProcessor

    model_name = args.model
    if model_name is None:
        from src.config import load_config

        model_name = load_config(args.config)["model"]["name"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rule(f"Loading {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device).eval()
    processor = IndicProcessor(inference=True)

    import transformers

    print(f"transformers : {transformers.__version__}")
    print(f"model class  : {type(model).__name__}")

    # ---------------------------------------------------------------- ON
    rule("A. Generation with KV cache ON (the fast path we want)")
    on_time, on_result = timed_generate(
        model, tokenizer, processor, SENTENCES, True, args.num_beams, args.max_length
    )
    cache_works = on_time is not None
    if cache_works:
        print(f"OK -- {on_time:.2f}s for {len(SENTENCES)} sentences")
        for src, out in zip(SENTENCES, on_result):
            print(f"  EN: {src}\n  MR: {out}")
    else:
        print(f"FAILED -- {on_result}")
        print("\nThis checkpoint's remote code predates the transformers Cache API.")

    # ---------------------------------------------------------------- OFF
    rule("B. Generation with KV cache OFF (the slow fallback)")
    off_time, off_result = timed_generate(
        model, tokenizer, processor, SENTENCES, False, args.num_beams, args.max_length
    )
    if off_time is not None:
        print(f"OK -- {off_time:.2f}s for {len(SENTENCES)} sentences")
    else:
        print(f"FAILED -- {off_result}")

    # ---------------------------------------------------------------- verdict
    rule("Verdict")
    if cache_works and off_time:
        speedup = off_time / on_time
        print(f"cache ON : {on_time:6.2f}s")
        print(f"cache OFF: {off_time:6.2f}s   ({speedup:.1f}x slower)")
        print()
        print("USE THIS CHECKPOINT. Set model.use_cache: true in the config and")
        print("restore evaluation.num_beams: 5 -- generation is cheap again.")
    elif cache_works:
        print("Cache works; the no-cache path errored, which is unusual but harmless.")
        print("USE THIS CHECKPOINT with model.use_cache: true.")
    else:
        print("Cache is unusable on this checkpoint. Options, best first:")
        print("  1. Use ai4bharat/indictrans2-en-indic-dist-200M (maintained code).")
        print("  2. Keep model.use_cache: false and accept slow generation. Training")
        print("     is unaffected -- teacher forcing never uses the cache.")
        print("  3. Downgrade to transformers<4.43, which still passes legacy tuples.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
