"""Answer the one question the rest of the pipeline depends on.

IndicTrans2 keeps two SentencePiece vocabularies -- one for the Latin-script
source, one for the Indic target. If we encode Marathi labels with the English
vocab, training still runs and the loss still falls; the model just learns
nothing useful. That failure is invisible until evaluation.

Run this FIRST, before writing or trusting any training run:

    python scripts/probe_tokenizer.py

It prints:
  * whether the model loads and translates at all (baseline smoke test)
  * which target-side tokenisation API this tokenizer exposes
  * a round-trip check: Marathi text -> ids -> text, which must come back intact
"""

from __future__ import annotations

import inspect
import sys

import torch

from pathlib import Path

# Running as `python scripts/x.py` puts scripts/ on sys.path, not the repo root,
# so `import src...` fails. Add the repo root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


MODEL = "prajdabre/rotary-indictrans2-en-indic-dist-200M"
SRC_LANG, TGT_LANG = "eng_Latn", "mar_Deva"

EN_SAMPLES = [
    "The weather is pleasant today.",
    "She submitted the report before the deadline.",
]
MR_SAMPLES = [
    "आज हवामान आल्हाददायक आहे.",
    "तिने मुदतीपूर्वी अहवाल सादर केला.",
]


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from IndicTransToolkit.processor import IndicProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rule(f"1. Loading {MODEL} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True).to(device).eval()
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    processor = IndicProcessor(inference=True)
    print(f"tokenizer class : {type(tokenizer).__name__}")
    print(f"model class     : {type(model).__name__}")
    print(f"pad / eos / bos : {tokenizer.pad_token_id} / {tokenizer.eos_token_id} / {tokenizer.bos_token_id}")
    print(f"decoder_start   : {model.config.decoder_start_token_id}")
    print(f"vocab size(s)   : {tokenizer.vocab_size}")

    # ---------------------------------------------------------------- 2
    rule("2. Baseline translation (does the untouched model work?)")
    prepared = processor.preprocess_batch(EN_SAMPLES, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    print("after preprocess_batch (note the language tags):")
    for line in prepared:
        print(f"  {line}")

    encoded = tokenizer(prepared, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded, num_beams=5, max_length=128, early_stopping=True,
            # This checkpoint's remote code predates the transformers Cache
            # API and crashes with it enabled. See modeling.disable_kv_cache().
            use_cache=False,
        )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    final = processor.postprocess_batch(decoded, lang=TGT_LANG)
    for src, out in zip(EN_SAMPLES, final):
        print(f"  EN : {src}\n  MR : {out}\n")

    # ---------------------------------------------------------------- 3
    rule("3. THE question: how do we encode target-side (Marathi) labels?")
    call_params = list(inspect.signature(tokenizer.__call__).parameters)
    print(f"tokenizer.__call__ params : {call_params}")
    print(f"has _switch_to_target_mode: {hasattr(tokenizer, '_switch_to_target_mode')}")
    print(f"has as_target_tokenizer   : {hasattr(tokenizer, 'as_target_tokenizer')}")
    print(f"accepts `src` flag        : {'src' in call_params}")
    print(f"accepts `text_target`     : {'text_target' in call_params}")

    from src.data import detect_label_strategy

    try:
        strategy = detect_label_strategy(tokenizer)
        print(f"\n  -> detected strategy: {strategy}")
    except RuntimeError as exc:
        print(f"\n  !! {exc}")
        return 1

    # ---------------------------------------------------------------- 4
    rule("4. Round-trip check (the actual correctness test)")
    print("Marathi -> ids -> Marathi. If the target vocab is being used, this")
    print("comes back readable. If the SOURCE vocab is used, expect heavy")
    print("fragmentation into single characters or <unk>.\n")

    from src.data import decode_targets, encode_labels

    label_ids = encode_labels(tokenizer, MR_SAMPLES, strategy, max_length=128)
    round_tripped = decode_targets(tokenizer, label_ids, strategy)

    ok = True
    for original, ids, recovered in zip(MR_SAMPLES, label_ids, round_tripped):
        chars = len(original)
        ratio = len(ids) / max(chars, 1)
        print(f"  original   : {original}")
        print(f"  n_tokens   : {len(ids)} for {chars} chars (tokens/char = {ratio:.2f})")
        print(f"  round-trip : {recovered}")
        # A healthy Indic SPM lands well under 1 token per character. Near or
        # above 1.0 means character-level fallback, i.e. the wrong vocabulary.
        if ratio > 0.9:
            print("  !! suspiciously high tokens/char -- likely the WRONG vocab")
            ok = False
        if recovered.strip() != original.strip():
            print("  ~  round-trip differs (normalisation may explain small diffs)")
        print()

    # ---------------------------------------------------------------- 5
    rule("5. Verdict")
    if ok:
        print("PASS -- wire this strategy into src/data.py and run:")
        print("    python -m src.train --smoke")
    else:
        print("FAIL -- do not start training. Inspect the tokenizer's source on")
        print("the Hub (it is downloaded under ~/.cache/huggingface/modules/) and")
        print("find the call that selects the target SentencePiece model.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
