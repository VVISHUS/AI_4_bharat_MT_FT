"""Samanantar -> filtered, tokenised en->mr training data.

Two things in here carry most of the engineering judgement for this project:

1. `build_filter_funnel` -- Samanantar is *bitext-mined*, not human-curated, so
   a meaningful fraction of pairs are misaligned, duplicated, or not actually
   Marathi. Every filter records what it dropped so the funnel can be reported.

2. `detect_label_strategy` -- IndicTrans2 keeps SEPARATE source and target
   SentencePiece vocabularies. Encoding Marathi labels with the English-side
   vocab is silently wrong: it trains, the loss falls, and the outputs are
   garbage. We identify the supported target-side path at runtime and raise if
   we cannot, rather than defaulting to the source vocab.
"""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from itertools import islice
from typing import Callable, Iterable

from datasets import Dataset, load_dataset

LOGGER = logging.getLogger(__name__)

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_WORD = re.compile(r"\S+")


# ---------------------------------------------------------------------------
# IndicProcessor adapter
# ---------------------------------------------------------------------------


class ProcessorAdapter:
    """Thin wrapper over IndicTransToolkit's IndicProcessor.

    The toolkit's API has shifted between releases (notably whether
    `preprocess_batch` accepts `is_target`), so we introspect rather than pin a
    version we cannot verify in this environment.
    """

    def __init__(self, inference: bool = False) -> None:
        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "IndicTransToolkit is required for IndicTrans2 preprocessing.\n"
                "  pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
            ) from exc

        self.processor = IndicProcessor(inference=inference)
        params = inspect.signature(self.processor.preprocess_batch).parameters
        self.supports_is_target = "is_target" in params
        if not self.supports_is_target:
            LOGGER.warning(
                "IndicProcessor.preprocess_batch has no `is_target`; the target "
                "side will be passed through unnormalised. Source-side language "
                "tags are unaffected, so this is a quality nit, not a bug."
            )

    def source(self, sentences: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Normalise and prepend the `<src_tag> <tgt_tag>` prefix IndicTrans2 expects."""
        return self.processor.preprocess_batch(
            sentences, src_lang=src_lang, tgt_lang=tgt_lang
        )

    def target(self, sentences: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Normalise the Indic side. No language tags -- targets carry none."""
        if not self.supports_is_target:
            return sentences
        return self.processor.preprocess_batch(
            sentences, src_lang=src_lang, tgt_lang=tgt_lang, is_target=True
        )

    def postprocess(self, sentences: list[str], lang: str) -> list[str]:
        """Undo entity placeholders / script mapping after generation."""
        return self.processor.postprocess_batch(sentences, lang=lang)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


@dataclass
class FilterStage:
    name: str
    reason: str
    predicate: Callable[[dict], bool]
    dropped: int = 0


@dataclass
class FilterReport:
    """The data funnel, in a form that drops straight into the README."""

    raw: int = 0
    kept: int = 0
    stages: list[FilterStage] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "| Stage | Why | Dropped | Remaining |",
            "| --- | --- | ---: | ---: |",
            f"| raw pool | rows streamed from Samanantar | - | {self.raw} |",
        ]
        remaining = self.raw
        for stage in self.stages:
            remaining -= stage.dropped
            lines.append(
                f"| {stage.name} | {stage.reason} | {stage.dropped} | {remaining} |"
            )
        pct = 100.0 * self.kept / self.raw if self.raw else 0.0
        lines.append(f"| **final** | | | **{self.kept}** ({pct:.1f}% of raw) |")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "kept": self.kept,
            "stages": [
                {"name": s.name, "reason": s.reason, "dropped": s.dropped}
                for s in self.stages
            ],
        }


def _devanagari_ratio(text: str) -> float:
    """Share of non-space characters that are Devanagari.

    Catches the common Samanantar failure where the 'Marathi' side is actually
    English, a URL, or a numeric table row.
    """
    stripped = [ch for ch in text if not ch.isspace()]
    if not stripped:
        return 0.0
    return len(_DEVANAGARI.findall(text)) / len(stripped)


def build_filter_funnel(filters: dict) -> list[FilterStage]:
    """Assemble the ordered filter stages from config.

    Order is chosen for report readability and cost: cheap structural checks
    first, content checks second, dedup last (dedup is stateful and grows with
    the number of rows kept).
    """
    min_words = filters["min_words"]
    max_words = filters["max_words"]
    max_ratio = filters["max_word_ratio"]
    min_deva = filters["min_devanagari_ratio"]

    stages: list[FilterStage] = [
        FilterStage(
            "non_empty",
            "either side blank after strip",
            lambda ex: bool(ex["_src"]) and bool(ex["_tgt"]),
        ),
        FilterStage(
            "length_bounds",
            f"word count outside [{min_words}, {max_words}] on either side",
            lambda ex: (
                min_words <= ex["_n_src"] <= max_words
                and min_words <= ex["_n_tgt"] <= max_words
            ),
        ),
        FilterStage(
            "length_ratio",
            f"longer/shorter side above {max_ratio}x (likely misaligned)",
            lambda ex: (
                max(ex["_n_src"], ex["_n_tgt"])
                / max(min(ex["_n_src"], ex["_n_tgt"]), 1)
                <= max_ratio
            ),
        ),
        FilterStage(
            "script_check",
            f"target under {min_deva:.0%} Devanagari (not Marathi)",
            lambda ex: _devanagari_ratio(ex["_tgt"]) >= min_deva,
        ),
    ]

    if filters.get("drop_copies", True):
        stages.append(
            FilterStage(
                "copy_pairs",
                "source identical to target (mining failure)",
                lambda ex: ex["_src"] != ex["_tgt"],
            )
        )
    return stages


def filter_stream(
    rows: Iterable[dict],
    src_column: str,
    tgt_column: str,
    filters: dict,
    target_size: int,
) -> tuple[list[dict], FilterReport]:
    """Run the funnel over a stream, stopping once `target_size` pairs survive.

    Streaming rather than materialising 3.63M rows keeps peak RAM inside a free
    Colab instance, which is the reason for the early exit.
    """
    stages = build_filter_funnel(filters)
    report = FilterReport(stages=stages)

    do_dedup = filters.get("dedup", True)
    dedup_on_source = filters.get("dedup_on_source", True)
    seen_pairs: set[int] = set()
    seen_sources: set[int] = set()
    dedup_stage = FilterStage("dedup", "exact duplicate pair", lambda ex: True)
    src_dedup_stage = FilterStage(
        "dedup_source",
        "source already seen (Samanantar has many targets per source)",
        lambda ex: True,
    )

    kept: list[dict] = []
    for row in rows:
        report.raw += 1
        src = (row.get(src_column) or "").strip()
        tgt = (row.get(tgt_column) or "").strip()
        example = {
            "_src": src,
            "_tgt": tgt,
            "_n_src": len(_WORD.findall(src)),
            "_n_tgt": len(_WORD.findall(tgt)),
        }

        rejected = False
        for stage in stages:
            if not stage.predicate(example):
                stage.dropped += 1
                rejected = True
                break
        if rejected:
            continue

        if do_dedup:
            pair_key = hash((src, tgt))
            if pair_key in seen_pairs:
                dedup_stage.dropped += 1
                continue
            seen_pairs.add(pair_key)
        if dedup_on_source:
            src_key = hash(src)
            if src_key in seen_sources:
                src_dedup_stage.dropped += 1
                continue
            seen_sources.add(src_key)

        kept.append({"src": src, "tgt": tgt})
        if len(kept) >= target_size:
            break

    if do_dedup:
        report.stages.append(dedup_stage)
    if dedup_on_source:
        report.stages.append(src_dedup_stage)
    report.kept = len(kept)
    return kept, report


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def detect_label_strategy(tokenizer) -> str:
    """Work out how to tokenise the Marathi side with the TARGET vocabulary.

    See the module docstring: getting this wrong is silent. We probe for the
    three known IndicTrans2 tokenizer APIs and raise if none is present rather
    than falling through to the source vocabulary.

    Returns one of: 'text_target' | 'src_flag' | 'as_target_tokenizer'
    """
    from transformers import PreTrainedTokenizerBase

    call_params = inspect.signature(tokenizer.__call__).parameters

    # Path A: the tokenizer implements the HF target-mode hooks, so the standard
    # `text_target=` kwarg routes through the target SPM.
    switch = getattr(type(tokenizer), "_switch_to_target_mode", None)
    base_switch = getattr(PreTrainedTokenizerBase, "_switch_to_target_mode", None)
    if switch is not None and switch is not base_switch and "text_target" in call_params:
        return "text_target"

    # Path B: older IndicTransTokenizer exposes an explicit `src` boolean.
    if "src" in call_params:
        return "src_flag"

    # Path C: the legacy context manager.
    if hasattr(tokenizer, "as_target_tokenizer"):
        return "as_target_tokenizer"

    raise RuntimeError(
        "Could not determine how to tokenise target-side text with this "
        "tokenizer. Run `python scripts/probe_tokenizer.py`, then wire the "
        "correct path into `encode_labels` before training. Do NOT fall back "
        "to the source vocabulary."
    )


def encode_labels(
    tokenizer, texts: list[str], strategy: str, max_length: int
) -> list[list[int]]:
    """Encode target text using the target-side vocabulary."""
    kwargs = {"truncation": True, "max_length": max_length}
    if strategy == "text_target":
        return tokenizer(text_target=texts, **kwargs)["input_ids"]
    if strategy == "src_flag":
        return tokenizer(texts, src=False, **kwargs)["input_ids"]
    if strategy == "as_target_tokenizer":
        with tokenizer.as_target_tokenizer():
            return tokenizer(texts, **kwargs)["input_ids"]
    raise ValueError(f"unknown label strategy {strategy!r}")


def encode_sources(tokenizer, texts: list[str], strategy: str, max_length: int) -> dict:
    """Encode source text using the source-side vocabulary."""
    kwargs = {"truncation": True, "max_length": max_length}
    if strategy == "src_flag":
        return tokenizer(texts, src=True, **kwargs)
    return tokenizer(texts, **kwargs)


def decode_targets(tokenizer, token_ids, strategy: str) -> list[str]:
    """Decode generated/label ids back to Marathi text.

    Mirror image of `encode_labels`: with two vocabularies, decoding target ids
    through the source vocab produces plausible-looking mojibake, so the older
    `src=` API needs the flag threaded through here too.
    """
    if strategy == "src_flag":
        try:
            return tokenizer.batch_decode(token_ids, src=False, skip_special_tokens=True)
        except TypeError:
            # Some builds accept the flag only at encode time.
            pass
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def build_tokenize_fn(
    tokenizer,
    processor: ProcessorAdapter,
    data_cfg: dict,
    strategy: str,
) -> Callable[[dict], dict]:
    """Batched map fn: raw pairs -> input_ids / attention_mask / labels."""
    src_lang = data_cfg["src_lang"]
    tgt_lang = data_cfg["tgt_lang"]
    max_src = data_cfg["max_source_length"]
    max_tgt = data_cfg["max_target_length"]

    def tokenize(batch: dict) -> dict:
        sources = processor.source(list(batch["src"]), src_lang, tgt_lang)
        targets = processor.target(list(batch["tgt"]), src_lang, tgt_lang)
        model_inputs = encode_sources(tokenizer, sources, strategy, max_src)
        model_inputs["labels"] = encode_labels(tokenizer, targets, strategy, max_tgt)
        return model_inputs

    return tokenize


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def load_samanantar_pairs(data_cfg: dict) -> tuple[list[dict], FilterReport]:
    """Stream Samanantar, filter it, return clean pairs plus the funnel report."""
    want = data_cfg["max_train_samples"] + data_cfg["valid_samples"]
    pool = int(want * data_cfg.get("raw_pool_multiplier", 1.6))
    streaming = data_cfg.get("streaming", True)

    dataset = load_dataset(
        data_cfg["dataset"],
        data_cfg["config"],
        split="train",
        streaming=streaming,
    )
    rows = islice(iter(dataset), pool) if streaming else dataset

    return filter_stream(
        rows,
        src_column=data_cfg["src_column"],
        tgt_column=data_cfg["tgt_column"],
        filters=data_cfg["filters"],
        target_size=want,
    )


def build_datasets(
    tokenizer,
    processor: ProcessorAdapter,
    data_cfg: dict,
    seed: int,
) -> tuple[Dataset, Dataset, FilterReport, str]:
    """Full path: load -> filter -> split -> tokenise."""
    pairs, report = load_samanantar_pairs(data_cfg)
    if not pairs:
        raise RuntimeError("Every pair was filtered out -- loosen the config filters.")

    dataset = Dataset.from_list(pairs).shuffle(seed=seed)
    n_valid = min(data_cfg["valid_samples"], max(1, len(dataset) // 10))
    valid_raw = dataset.select(range(n_valid))
    train_raw = dataset.select(range(n_valid, len(dataset)))

    strategy = detect_label_strategy(tokenizer)
    LOGGER.info("Target-side tokenisation strategy: %s", strategy)

    tokenize = build_tokenize_fn(tokenizer, processor, data_cfg, strategy)
    columns = train_raw.column_names
    train = train_raw.map(
        tokenize, batched=True, remove_columns=columns, desc="tokenise[train]"
    )
    valid = valid_raw.map(
        tokenize, batched=True, remove_columns=columns, desc="tokenise[valid]"
    )
    return train, valid, report, strategy
