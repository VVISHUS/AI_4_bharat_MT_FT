# Fine-tuning rotary-IndicTrans2-200M for English → Marathi

Fine-tunes [`prajdabre/rotary-indictrans2-en-indic-dist-200M`](https://huggingface.co/prajdabre/rotary-indictrans2-en-indic-dist-200M)
on [`ai4bharat/samanantar`](https://huggingface.co/datasets/ai4bharat/samanantar) (`mr`),
evaluated on IN22-Gen.

> **Status:** `<!-- fill in after your run -->`

---

## Quick start

```bash
pip install -r requirements.txt
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

# 1. Settle the tokenizer question before anything else (see "The two-vocabulary trap")
python scripts/probe_tokenizer.py

# 2. Prove the label pipeline works by overfitting 32 pairs
python -m src.train --config configs/finetune_en_mr.yaml --smoke

# 3. Baseline, before training — you need a "before" to claim a delta
python -m src.evaluate --config configs/finetune_en_mr.yaml --base-only --output outputs/baseline

# 4. The real run
python -m src.train --config configs/finetune_en_mr.yaml

# 5. Compare
python -m src.evaluate --config configs/finetune_en_mr.yaml \
    --checkpoint outputs/rotary-it2-en-mr/final \
    --output outputs/rotary-it2-en-mr
```

Or run [`notebooks/colab_finetune_indictrans2_marathi.ipynb`](notebooks/colab_finetune_indictrans2_marathi.ipynb)
end to end on a Colab T4.

Any config leaf can be overridden without editing YAML:

```bash
python -m src.train --set training.learning_rate=5e-5 peft.enabled=false
```

---

## Repository layout

```
configs/finetune_en_mr.yaml   all hyperparameters + filter thresholds, commented
src/config.py                 YAML loading, dotted CLI overrides, run snapshotting
src/data.py                   Samanantar streaming, the filter funnel, tokenisation
src/modeling.py               model/tokenizer loading, dtype selection, LoRA attachment
src/train.py                  Seq2SeqTrainer setup; --smoke overfit mode
src/evaluate.py               base vs fine-tuned on IN22-Gen; chrF++ / BLEU
scripts/probe_tokenizer.py    the pre-flight correctness check
tests/test_filters.py         filter-funnel tests; no GPU, no network, no deps
notebooks/…ipynb              annotated Colab driver
```

The filter tests stub the `datasets` import, so they run anywhere:

```bash
python tests/test_filters.py        # or: python -m pytest tests/ -v
```

---

## Approach

### Why MT, and why this checkpoint

MT was chosen over ASR and TTS for one reason: it has the shortest path from
"nothing" to "a verified end-to-end run", and the assignment is explicitly graded
on getting a run working rather than on metrics. No audio decoding, no
resampling, no vocoder, and an objective metric that takes seconds to compute.

The `en-indic` direction matches Samanantar's layout directly — its `src` column
is English and `tgt` is the Indic language, so `eng_Latn → mar_Deva` needs no
inversion.

### The document-level mismatch (read this first)

The model card states these rotary checkpoints are **"primarily built and tested
for document-level and long-context translations."** Samanantar is a
*sentence-level* corpus. These are not aligned.

This was a deliberate, documented choice rather than an oversight:

- The assignment specifies a Marathi dataset from AIKosh or HuggingFace, and
  Samanantar is the canonical large Marathi parallel corpus.
- Rotary position embeddings do not *require* long inputs; they generalise
  across lengths. Fine-tuning on short segments is valid, it simply exercises
  none of the reason this variant exists.
- The honest consequence: this run cannot demonstrate the checkpoint's actual
  advantage over the sinusoidal baseline.

With more time, the right corpus is **BPCC-doc** or another document-aligned
source, training on multi-sentence windows with document context carried across
segment boundaries. That would put the variant's long-context capability under
actual load.

### The two-vocabulary trap

IndicTrans2 keeps **separate source and target SentencePiece vocabularies**.
Encoding Marathi labels with the English-side vocabulary produces a failure that
is entirely silent:

1. The Marathi text shatters into character-level pieces and `<unk>`.
2. Training runs without error.
3. The loss decreases smoothly and looks healthy.
4. The model learns nothing, and you find out at evaluation.

`src/data.py:detect_label_strategy` therefore probes for the three known
target-side APIs (`text_target=`, a `src=` boolean, `as_target_tokenizer()`) and
**raises rather than falling back to the source vocabulary**. The same flag is
threaded through decoding in `decode_targets`, because the mirror-image bug
exists on the way out.

`scripts/probe_tokenizer.py` verifies this empirically with a Marathi round-trip
and a tokens-per-character check — a healthy Indic SPM sits well under 1.0;
anything near it means character fallback, i.e. the wrong vocabulary.

**Result on this checkpoint:** `<!-- paste the probe output -->`

### Data filtering

Samanantar is **bitext-mined**, not human-curated, so misalignment, duplication
and wrong-language rows occur at a meaningful rate. Every filter targets a
specific observed failure and reports what it removed:

| Filter | Catches |
| --- | --- |
| `non_empty` | blank sides after strip |
| `length_bounds` | fragments and headers; runaway rows that inflate padding |
| `length_ratio` | misaligned pairs where one side isn't a translation of the other |
| `script_check` | "Marathi" rows that are actually English, URLs or number tables |
| `copy_pairs` | `src == tgt`, i.e. the miner gave up |
| `dedup` / `dedup_source` | exact duplicates; Samanantar has many targets per source |

Loading is **streaming with early exit** — 3.63M rows are never materialised,
which keeps peak RAM inside a free Colab instance.

**Funnel from the actual run:**

```
<!-- paste outputs/rotary-it2-en-mr/filter_report.md -->
```

### Training configuration

| Choice | Value | Reasoning |
| --- | --- | --- |
| Subset | 120k pairs | Deliberate compute budget, not a limitation. ~3.7k optimiser steps — enough to move the model, short enough to finish and still write this up. |
| LoRA | r=16, α=32 | ~1% trainable params, so optimiser state stays small and batch 16 fits a T4 with headroom. Set `peft.enabled=false` to full-fine-tune and compare. |
| LR | 1e-4 | LoRA tolerates roughly 10× the full-FT rate since only adapters move. |
| Effective batch | 32 | 16 × 2 gradient accumulation. MT benefits from larger batches; this is the most available cheaply. |
| Label smoothing | 0.1 | Standard for NMT; reduces overconfidence. |
| `group_by_length` | true | Batches similar-length sequences, cutting padding waste on variable-length MT data. |
| Schedule | cosine, 3% warmup | Warmup matters — a cold high LR on an already-strong pretrained MT model degrades it quickly. |
| Precision | auto | bf16 on Ampere+, fp16 on Turing. T4 has **no** bf16 support and asking for it doesn't error, so `src/modeling.py` checks instead of hardcoding. |

### Evaluation

Scored on **IN22-Gen** (AI4Bharat's own benchmark), not a held-out Samanantar
split. Evaluating on held-out rows of a mined corpus measures how well the model
fits the miner's noise, which isn't the question. `src/evaluate.py` falls back to
FLORES automatically if the Hub config name has drifted.

**chrF++ is the headline metric.** BLEU on a morphologically rich target with a
single reference is noisy and rewards surface n-gram overlap; it's reported
because it's expected, not because it's informative here.

Base and fine-tuned are decoded with identical generation settings so the
comparison isolates the weights.

| Model | chrF++ | BLEU |
| --- | ---: | ---: |
| Base | `<!-- -->` | `<!-- -->` |
| Fine-tuned | `<!-- -->` | `<!-- -->` |
| Δ | `<!-- -->` | `<!-- -->` |

`<!-- If fine-tuned < base, say so and give the mechanism. The likely one:
IndicTrans2 was trained on BPCC, which already subsumes Samanantar plus far more
curated data, so fine-tuning on 120k mined pairs narrows the distribution.
Supporting evidence: the gap should be larger on IN22-Gen (out of domain) than
on Samanantar validation (in domain). A candid analysis of a regression is a
better result than a lucky number. -->`

---

## Order of operations, and why

The pipeline is deliberately ordered so that the cheapest checks fail first:

1. **Probe the tokenizer** — settles the two-vocabulary question before any
   code depends on the answer.
2. **Overfit 32 pairs** — a 200M model should memorise them almost completely.
   If loss doesn't collapse below 1.0, the label pipeline is broken. Two minutes
   here prevents a wasted multi-hour run; `src/train.py` exits non-zero if it
   fails.
3. **Inspect the data funnel** — read thirty real rows before trusting any
   threshold.
4. **Baseline eval before training** — establishes the "before" number and
   exercises the full inference path while there's still time to fix it.
5. **Train.**
6. **Eval + qualitative diff** — metrics hide truncation, repetition and dropped
   entities.

---

## Challenges

**Local hardware was unusable for this.** The development machine has a 4GB
RTX 3050 with a CPU-only torch build on Python 3.13. 200M parameters under Adam
needs ~3.2GB before activations, and `IndicTransToolkit` has a C build step that
wants Python ≤3.12. Rather than spend hours on a CUDA reinstall and wheel
compatibility, the repo is authored locally and trained on Colab/Kaggle (Python
3.11, CUDA preinstalled, 16GB). `src/modeling.py` detects the device and dtype
at runtime so the same code runs in both places.

`<!-- Add the ones you actually hit. Strong candidates:
  - transformers version vs. the Hub's remote code
  - IndicProcessor API drift (is_target)
  - IN22-Gen config naming on the Hub
  - OOM / throughput tuning
  - anything the smoke test caught
-->`

---

## What I'd do with more time

- **Full BPCC** rather than a 120k Samanantar subset, with quality-based
  filtering (LaBSE cosine) instead of only structural heuristics.
- **Document-level training** on BPCC-doc, to actually exercise what these
  rotary checkpoints were built for.
- **LoRA vs. full fine-tuning** as a controlled comparison at equal step count.
- **Human evaluation** on a sample. chrF++ is a proxy, and for a
  morphologically rich target a cheap proxy.
- **Domain-targeted evaluation** — split IN22-Gen by domain to see where the
  fine-tune helps and where it regresses, rather than reporting one aggregate.

---

## Artifacts

Checkpoints, logs and outputs: `<!-- Google Drive link -->`
