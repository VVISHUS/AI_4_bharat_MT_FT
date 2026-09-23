"""Fine-tune rotary-IndicTrans2-dist-200M on Samanantar en->mr.

Usage
-----
    # 1. sanity: overfit 32 pairs. Loss MUST collapse toward zero.
    python -m src.train --config configs/finetune_en_mr.yaml --smoke

    # 2. the real run
    python -m src.train --config configs/finetune_en_mr.yaml

    # 3. anything can be overridden without editing the YAML
    python -m src.train --set training.learning_rate=5e-5 peft.enabled=false

The `--smoke` path exists because the expensive failure mode in seq2seq
fine-tuning is a label pipeline that is subtly wrong: training runs, loss
decreases, and the model has learned nothing. Overfitting a handful of examples
is the cheapest possible test that gradients actually reach the target text.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from src.config import base_arg_parser, load_config, save_config
from src.data import ProcessorAdapter, build_datasets, decode_targets
from src.modeling import attach_lora, count_parameters, load_model, load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("train")


def build_compute_metrics(tokenizer, strategy: str):
    """chrF++ on the in-domain validation split, for a during-training signal.

    This is NOT the headline number -- Samanantar validation is in-domain and
    mined, so it flatters the model. evaluate.py reports the real figure on the
    held-out IN22-Gen benchmark.
    """
    import sacrebleu

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Trainer pads labels with -100 to mask loss; those ids are not in the
        # vocab and will crash the decoder if left in.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        decoded_preds = decode_targets(tokenizer, preds, strategy)
        decoded_labels = decode_targets(tokenizer, labels, strategy)

        chrf = sacrebleu.corpus_chrf(
            [p.strip() for p in decoded_preds],
            [[l.strip() for l in decoded_labels]],
            word_order=2,  # chrF++ rather than plain chrF
        )
        return {"chrf++": round(chrf.score, 3)}

    return compute_metrics


class DataCollatorWithDecoderInputs(DataCollatorForSeq2Seq):
    """Seq2seq collator that also builds `decoder_input_ids` from `labels`.

    Most HF seq2seq models derive the decoder input by shifting the labels one
    position right, either inside `forward` or via a
    `prepare_decoder_input_ids_from_labels` method that `DataCollatorForSeq2Seq`
    calls automatically. This checkpoint's remote modeling code does **neither**,
    so training dies with:

        ValueError: You have to specify either decoder_input_ids or
                    decoder_inputs_embeds

    Inference is unaffected because `generate()` constructs decoder inputs on
    its own -- which is why the probe passed and training did not.

    The shift is the standard fairseq/BART one:

        labels            = [ y1, y2, ..., yn, </s> ]
        decoder_input_ids = [ <start>, y1, y2, ..., yn ]

    with `<start>` = `decoder_start_token_id` (2, i.e. </s>, for IndicTrans2).
    Padding positions in `labels` carry -100 so they are excluded from the loss;
    those must become real pad ids here, because -100 is not a valid embedding
    index and would index-error the decoder's embedding lookup.
    """

    def __init__(self, *args, decoder_start_token_id: int, pad_token_id: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_start_token_id = decoder_start_token_id
        self.label_pad_id = pad_token_id

    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors=return_tensors)
        if "labels" not in batch or "decoder_input_ids" in batch:
            return batch

        labels = batch["labels"]
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = self.decoder_start_token_id
        # -100 is a loss-masking sentinel, never a token. Must not reach nn.Embedding.
        shifted.masked_fill_(shifted == -100, self.label_pad_id)
        batch["decoder_input_ids"] = shifted
        return batch


def build_training_args(cfg: dict, output_dir: Path, smoke: bool) -> Seq2SeqTrainingArguments:
    tcfg = cfg["training"]
    use_cuda = torch.cuda.is_available()
    # bf16 where the hardware allows it (Ampere+), fp16 on Turing (T4).
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    fp16 = use_cuda and not bf16

    kwargs = dict(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=float(tcfg["learning_rate"]),
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        label_smoothing_factor=tcfg["label_smoothing_factor"],
        max_grad_norm=tcfg["max_grad_norm"],
        logging_steps=tcfg["logging_steps"],
        save_total_limit=tcfg["save_total_limit"],
        predict_with_generate=tcfg["predict_with_generate"],
        generation_num_beams=tcfg["generation_num_beams"],
        generation_max_length=tcfg["generation_max_length"],
        group_by_length=tcfg["group_by_length"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        report_to=tcfg["report_to"],
        fp16=fp16,
        bf16=bf16,
        seed=cfg["seed"],
        # Two dataloader workers is the sweet spot on Colab; more contends with
        # the single vCPU pair you actually get.
        dataloader_num_workers=2,
        remove_unused_columns=True,
        logging_first_step=True,
    )

    if smoke:
        # Overfit mode: no eval, log every step, many passes over a tiny set.
        kwargs.update(
            num_train_epochs=30,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            warmup_ratio=0.0,
            logging_steps=5,
            save_strategy="no",
            eval_strategy="no",
            group_by_length=False,
            predict_with_generate=False,
        )
    else:
        kwargs.update(
            eval_strategy="steps",
            eval_steps=tcfg["eval_steps"],
            save_strategy="steps",
            save_steps=tcfg["save_steps"],
            load_best_model_at_end=True,
            metric_for_best_model="chrf++",
            greater_is_better=True,
        )

    # `evaluation_strategy` was renamed to `eval_strategy`; support both so this
    # runs on whatever transformers version Colab ships this week.
    try:
        return Seq2SeqTrainingArguments(**kwargs)
    except TypeError as exc:
        if "eval_strategy" not in str(exc):
            raise
        if "eval_strategy" in kwargs:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return Seq2SeqTrainingArguments(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = base_arg_parser("Fine-tune IndicTrans2 (rotary) on Samanantar en->mr.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Overfit 32 pairs to prove the label pipeline is wired correctly.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["seed"])

    output_dir = Path(cfg["training"]["output_dir"])
    if args.smoke:
        output_dir = output_dir.with_name(output_dir.name + "-smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir)

    LOGGER.info("Loading tokenizer and model: %s", cfg["model"]["name"])
    tokenizer = load_tokenizer(cfg["model"])
    model = load_model(cfg["model"])
    base_config = model.config  # captured before the PEFT wrap

    # ---- data -------------------------------------------------------------
    if args.smoke:
        # Deliberately tiny, and NOT deduped against itself: we want the model
        # to memorise these 32 pairs.
        cfg["data"] = {**cfg["data"], "max_train_samples": 32, "valid_samples": 8}

    processor = ProcessorAdapter(inference=False)
    train_ds, valid_ds, report, strategy = build_datasets(
        tokenizer, processor, cfg["data"], seed=cfg["seed"]
    )

    LOGGER.info("Filter funnel:\n%s", report.to_markdown())
    (output_dir / "filter_report.md").write_text(report.to_markdown(), encoding="utf-8")
    (output_dir / "filter_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    LOGGER.info("train=%d  valid=%d  label_strategy=%s", len(train_ds), len(valid_ds), strategy)

    # Sanity print: if labels are empty or absurdly short, stop here.
    sample = train_ds[0]
    LOGGER.info(
        "sample lengths -> input_ids=%d labels=%d",
        len(sample["input_ids"]),
        len(sample["labels"]),
    )
    if len(sample["labels"]) < 2:
        raise RuntimeError(
            "Labels are empty/degenerate -- target-side tokenisation is wrong. "
            "Run scripts/probe_tokenizer.py."
        )

    # ---- model ------------------------------------------------------------
    if cfg["peft"]["enabled"]:
        model = attach_lora(model, cfg["peft"])
    trainable, total = count_parameters(model)
    LOGGER.info("trainable %s / %s params (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # Cache stays off throughout: training never uses it (teacher forcing), and
    # this checkpoint's remote code cannot handle the modern Cache objects that
    # `predict_with_generate` evaluation would otherwise create.
    model.config.use_cache = False

    # Read the special ids off the *base* config: after the PEFT wrap, attribute
    # proxying works but is easier to reason about if we capture them up front.
    decoder_start_token_id = base_config.decoder_start_token_id
    pad_token_id = base_config.pad_token_id
    if decoder_start_token_id is None:
        raise RuntimeError(
            "decoder_start_token_id is None -- the decoder has no token to begin "
            "from. Check what scripts/probe_tokenizer.py reported."
        )
    LOGGER.info(
        "decoder_start_token_id=%s pad_token_id=%s", decoder_start_token_id, pad_token_id
    )

    collator = DataCollatorWithDecoderInputs(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
        decoder_start_token_id=decoder_start_token_id,
        pad_token_id=pad_token_id,
    )

    training_args = build_training_args(cfg, output_dir, args.smoke)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None if args.smoke else valid_ds,
        data_collator=collator,
        compute_metrics=None if args.smoke else build_compute_metrics(tokenizer, strategy),
    )

    LOGGER.info("Starting %s run", "SMOKE" if args.smoke else "full")
    result = trainer.train()

    if args.smoke:
        final_loss = result.training_loss
        LOGGER.info("Smoke run final training loss: %.4f", final_loss)
        if final_loss > 1.0:
            LOGGER.error(
                "Loss did not collapse on 32 examples. The label pipeline is "
                "almost certainly wrong -- do NOT start the full run."
            )
            return 1
        LOGGER.info("Smoke test PASSED: gradients reach the target text.")
        return 0

    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    (output_dir / "train_metrics.json").write_text(
        json.dumps(result.metrics, indent=2), encoding="utf-8"
    )
    # Record which tokenisation path was used -- evaluate.py needs the same one.
    (output_dir / "label_strategy.json").write_text(
        json.dumps({"strategy": strategy}), encoding="utf-8"
    )
    LOGGER.info("Done. Artifacts in %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
