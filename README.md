# IndicTrans2 English → Marathi fine-tuning

LoRA fine-tuning of [`ai4bharat/indictrans2-en-indic-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M)
on [`ai4bharat/samanantar`](https://huggingface.co/datasets/ai4bharat/samanantar) (`mr`),
scored on FLORES-200 and on a held-out slice of the training corpus.

A plain-language account of the decisions, the problems hit along the way, and
what I would change is in [APPROACH.md](APPROACH.md).

---

## Quick start

```bash
pip install -r requirements.txt
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
export HF_TOKEN=...        # the checkpoint and FLORES are both gated (auto-approve)

# 1. confirm how the tokenizer encodes target-side text
python scripts/probe_tokenizer.py

# 2. confirm whether the KV cache is usable on this checkpoint
python scripts/check_kv_cache.py

# 3. overfit 32 pairs to prove the label pipeline is wired correctly
python -m src.train --smoke

# 4. baseline, before training
python -m src.evaluate --base-only --output outputs/baseline

# 5. train
python -m src.train

# 6. evaluate, out of domain and in domain
python -m src.evaluate --checkpoint outputs/indictrans2-en-mr/final \
                       --output outputs/indictrans2-en-mr
python -m src.evaluate --checkpoint outputs/indictrans2-en-mr/final \
                       --output outputs/indictrans2-en-mr --in-domain
```

[`notebooks/colab_finetune_indictrans2_marathi.ipynb`](notebooks/colab_finetune_indictrans2_marathi.ipynb)
drives the same pipeline on a Colab T4.

Any config leaf can be overridden without editing YAML:

```bash
python -m src.train --set training.learning_rate=5e-5 peft.enabled=false
```

---

## Layout

```
configs/finetune_en_mr.yaml   hyperparameters and filter thresholds
src/config.py                 YAML loading, dotted CLI overrides, run snapshotting
src/data.py                   streaming, the filter funnel, tokenisation
src/modeling.py               model/tokenizer loading, dtype, LoRA attachment
src/train.py                  collator, Trainer setup, --smoke overfit mode
src/evaluate.py               base vs fine-tuned; chrF++ / BLEU
scripts/probe_tokenizer.py    target-vocabulary pre-flight check
scripts/check_kv_cache.py     KV-cache compatibility and cost measurement
```

---

## Approach

### Two SentencePiece vocabularies

IndicTrans2 keeps **separate source and target vocabularies** (`model.SRC` /
`model.TGT`, 759 KB and 3.26 MB respectively — the target side covers 22
languages). Encoding Marathi labels with the English-side vocabulary fails
silently: the text fragments into `<unk>`, training runs, the loss falls
smoothly, and the model learns nothing. The failure is invisible until decoding.

`src/data.py:detect_label_strategy` therefore probes for the three known
target-side APIs (`text_target=`, an `src=` boolean, `as_target_tokenizer()`)
and **raises rather than falling back to the source vocabulary**. The same flag
is threaded through `decode_targets`, since the mirror-image bug exists on the
way out. `scripts/probe_tokenizer.py` verifies it empirically with a Marathi
round-trip and a tokens-per-character check: a healthy Indic SPM sits well under
1.0, and anything near it indicates character fallback.

This checkpoint resolves to `text_target`.

### Data filtering

Samanantar is **bitext-mined**, not human-curated, so misalignment, duplication
and wrong-language rows occur at a measurable rate. Each filter targets one
observed failure, and the funnel is reported per run:

| Stage | Dropped | Remaining |
| --- | ---: | ---: |
| raw pool | — | 137,428 |
| `non_empty` — blank after strip | 0 | 137,428 |
| `length_bounds` — outside [3, 80] words | 5,166 | 132,262 |
| `length_ratio` — longer/shorter above 2.5× | 2,036 | 130,226 |
| `script_check` — target under 55% Devanagari | 106 | 130,120 |
| `copy_pairs` — source identical to target | 0 | 130,120 |
| `dedup` — exact duplicate pair | 0 | 130,120 |
| `dedup_source` — source already seen | 9,120 | **121,000** |

**88.0% retained.** Two results worth noting: exact duplicate *pairs* are zero,
suggesting Samanantar was already deduplicated upstream at pair level, while
`dedup_source` is the largest single filter — the corpus genuinely carries
multiple Marathi translations per English source.

Loading streams with early exit, so the remaining ~3.5M rows are never
downloaded. The subset is 121k pairs, **3.3% of the corpus**, a deliberate
compute budget rather than a limitation.

### Training

| Choice | Value | Reasoning |
| --- | --- | --- |
| LoRA | r=16, α=32 | 6,488,064 / 218,264,576 params trainable (2.97%). Fits a T4 with headroom; `--set peft.enabled=false` for full FT |
| LR | 1e-4 | LoRA tolerates ~10× the full-FT rate, since only adapters move |
| Effective batch | 32 | 16 × 2 gradient accumulation |
| Label smoothing | 0.1 | Standard for NMT. Puts a **floor** on the loss at ~1.363 for this vocab |
| `group_by_length` | true | Batches similar lengths, cutting padding waste |
| Schedule | cosine, 3% warmup | Warmup protects the pretrained weights from a cold high-LR shock |
| Precision | auto | bf16 on Ampere+, fp16 on Turing (T4 has no bf16 support) |

3,750 steps, 52m40s on a T4.

### Evaluation

Headline metric is **chrF++ on a human-curated, out-of-domain benchmark**.
`--in-domain` additionally scores the held-out Samanantar split, and the gap
between the two is the informative quantity: a fine-tune that improves in-domain
while flat or worse out-of-domain has fitted the mined corpus's quirks rather
than learned better Marathi.

BLEU is reported alongside because it is expected, but with a single reference
and a morphologically rich target it is the weaker signal.

Base and fine-tuned decode with identical settings, so the comparison isolates
the weights.

500 sentences each, beam 5, identical decoding for base and fine-tuned.

**FLORES-200 `devtest`** — human-translated, out of domain:

| Model | chrF++ | BLEU |
| --- | ---: | ---: |
| Base | **50.72** | **15.15** |
| Fine-tuned | 48.72 | 13.44 |
| Δ | −2.00 | −1.71 |

**Held-out Samanantar** — split off before training and never seen:

| Model | chrF++ | BLEU |
| --- | ---: | ---: |
| Base | 38.70 | 7.78 |
| Fine-tuned | **39.21** | **8.48** |
| Δ | +0.51 | +0.70 |

Two things fall out of this pair of tables.

First, the signs are opposite. Fine-tuning moved the in-domain score up and the
out-of-domain score down. That is the signature of fitting the corpus rather than
improving the translation: the adapter learned something real about Samanantar,
and what it learned does not transfer. Had I only reported the in-domain number
this would read as a successful fine-tune.

Second, the magnitudes are lopsided — +0.51 in domain against −2.00 out of it.
The model gave up four times as much general quality as it gained on the corpus
it was trained on, so this is not even a favourable trade for someone who only
cares about Samanantar-like text.

The baseline gap between the two benchmarks is worth noting separately: the same
untouched model scores **12 points lower** against Samanantar references than
against FLORES. That is a statement about the references, not the model — mined
web text disagrees with correct translations often enough to depress the score.
It is direct evidence that the training corpus is noisier than the benchmark,
which is the same fact the opposite-signed deltas are pointing at.

---

## Results

**Fine-tuning made the model worse by 2.0 chrF++.** That is the expected outcome
here, and the reason is visible in the loss curve.

Training loss over one epoch:

```
step   50:  4.18
step  200:  3.83
step 3750:  3.74      (final train_loss 3.747)
```

Loss falls sharply through warmup and then flattens, moving 0.09 over the final
3,550 steps. Three factors explain this, and they compound:

1. **The data is inside the model's training distribution.** IndicTrans2 was
   trained on BPCC, which incorporates Samanantar, and Marathi is one of its 22
   core languages. This is continued training on material the model has already
   seen, not the addition of a new language.
2. **LoRA at r=16 moves 2.97% of the weights.**
3. **Label smoothing floors the loss at 1.363**, so 3.74 is not convergence to
   an optimum — it reflects how little signal was available to extract.

A flat curve is the expected shape for this combination, not a defect. The
regression follows from it directly: with little to gain in-distribution, the
2.97% of weights that did move specialised toward a mined corpus with a
measurable misalignment rate, and that specialisation costs accuracy on clean
out-of-domain text. The two evaluation tables above are what that sentence looks
like when measured — the in-domain score goes up, the out-of-domain score goes
down, and the second effect is the larger of the two.

For a run with real headroom the corpus would need to sit outside BPCC:
**Bhili** or another genuinely low-resource language, or a Marathi domain the
base model handles poorly.

---

## Order of operations

The pipeline is sequenced so the cheapest checks fail first:

1. **Probe the tokenizer** — settles the two-vocabulary question before anything
   depends on the answer.
2. **Check the KV cache** — measures, rather than assumes, what generation costs.
3. **Overfit 32 pairs** — with smoothing and LoRA disabled so zero is reachable,
   loss must fall below 0.5. `src/train.py` exits non-zero otherwise.
4. **Baseline evaluation** — establishes the "before" number and exercises the
   full inference path while there is still time to fix it.
5. **Train.**
6. **Evaluate and diff qualitatively** — metrics hide truncation, repetition and
   dropped entities.

---

## Notes on the checkpoint and environment

**The KV cache cannot be enabled.** The published IndicTrans2 remote modeling
code targets the pre-4.43 transformers API, where `generate()` passed
`past_key_values=None` on the first decoding step. Under transformers 4.56 an
empty `EncoderDecoderCache` is passed instead, and

```python
past_key_values[0][0].shape[2] if past_key_values is not None else 0
```

takes the wrong branch and raises on `NoneType.shape`. Community forks of the
checkpoint behave identically.

Cost is modest and measured rather than assumed — `scripts/check_kv_cache.py`
reports **0.77s for 4 sentences at beam 5**, ~0.19 s/sentence, so a 500-sentence
benchmark decodes in ~95s. Training is unaffected: teacher forcing is a single
parallel pass and never decodes step by step.

**The model does not derive `decoder_input_ids` from labels.** Most HF seq2seq
models shift labels right inside `forward`, or expose
`prepare_decoder_input_ids_from_labels` for `DataCollatorForSeq2Seq` to call.
This one does neither, so training fails with
`ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds`
while inference works fine (`generate()` builds them itself).
`DataCollatorWithDecoderInputs` in `src/train.py` performs the standard
fairseq/BART shift, mapping `-100` label padding back to real pad ids so it
never reaches the embedding lookup.

**Evaluation during training is disabled** (`eval_during_training: false`, which
also passes `eval_dataset=None`). With the cache off, a generation-based eval
pass over 1,000 samples costs roughly as much as the entire training run, and the
validation split is drawn from the same mined corpus as the training data.

**PEFT and torchao conflict on Colab.** PEFT's LoRA dispatcher probes for
optional quantization backends and raises on torchao below 0.16 rather than
treating it as unavailable; Colab preinstalls 0.10. `pip uninstall -y torchao`
resolves it, and `src/modeling.py` converts the error into that advice.

**Benchmark loading is defensive.** IN22-Gen has been repackaged as parquet, so
the per-pair config names on its card may not resolve, and its split is `gen`
rather than `test`. FLORES is gated and cannot serve as a silent fallback. The
loader tries a list of candidate shapes and reports which succeeded.

