"""Model + tokenizer construction, dtype selection, and LoRA attachment.

Split out from train.py so the notebook, the probe script and evaluate.py all
load the model exactly the same way.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


def pick_dtype(requested: str = "auto") -> torch.dtype:
    """Choose a compute dtype the current GPU actually supports.

    A free-tier T4 is Turing: it has fp16 tensor cores but NOT bf16. Asking for
    bf16 there does not error, it silently runs slow or destabilises. So we
    check rather than hardcode.
    """
    if requested not in {"auto", None}:
        return getattr(torch, requested)
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_tokenizer(model_cfg: dict):
    """IndicTrans2 ships a custom tokenizer class, hence trust_remote_code."""
    return AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )


def load_model(model_cfg: dict, dtype: torch.dtype | None = None):
    """Load the seq2seq model.

    Note we do NOT pass torch_dtype for training. Mixed precision is handled by
    the Trainer (fp16/bf16 flags), which keeps fp32 master weights; loading the
    weights in fp16 *and* training would give us no master copy and unstable
    updates. Inference paths pass an explicit dtype.
    """
    kwargs = {"trust_remote_code": model_cfg.get("trust_remote_code", True)}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    return AutoModelForSeq2SeqLM.from_pretrained(model_cfg["name"], **kwargs)


def resolve_lora_targets(model, requested: list[str]) -> list[str]:
    """Keep only the requested module names that this checkpoint actually has.

    The rotary variant renames parts of the attention block relative to vanilla
    IndicTrans2. Passing a name that does not exist makes PEFT raise a fairly
    opaque error, so we intersect against the real module names first and fall
    back to every nn.Linear leaf if nothing matches.
    """
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    matched = [name for name in requested if name in present]
    if matched:
        LOGGER.info("LoRA target modules: %s", matched)
        return matched

    linear_leaves = sorted(
        {
            name.split(".")[-1]
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and "lm_head" not in name
        }
    )
    LOGGER.warning(
        "None of %s exist in this checkpoint. Falling back to all Linear "
        "leaves: %s",
        requested,
        linear_leaves,
    )
    return linear_leaves


def attach_lora(model, peft_cfg: dict):
    """Wrap the model in LoRA adapters and report the trainable fraction."""
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=peft_cfg["r"],
        lora_alpha=peft_cfg["alpha"],
        lora_dropout=peft_cfg["dropout"],
        target_modules=resolve_lora_targets(model, peft_cfg["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def count_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
