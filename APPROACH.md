# How I built this, and why

A plain-language account of the decisions behind this project — what I chose,
what went wrong, and what I'd do differently.

---

## 1. Why machine translation

There were three options: translation, speech recognition, and text-to-speech.
I picked translation.

Speech recognition means handling audio: resampling every clip to a common rate,
filtering out files that are too long, cleaning up transcripts. Text-to-speech is
harder still — you need many hours of clean recordings from a single speaker, and
you can't tell whether training worked until you sit and listen to the output. A
half-trained speech model sounds like static, which is indistinguishable from a
bug in your code.

Translation is text in, text out. Nothing to decode, nothing to listen to, and
you can score the result in seconds. That meant I could spend my time on the
parts that actually matter — data quality and getting the training loop right —
rather than on audio plumbing.

---

## 2. Picking the data, and the mistake I'd undo

I used Samanantar, the standard large English–Marathi corpus, 3.63 million
sentence pairs.

This turned out to be the weakest decision in the project, and it's worth being
direct about why.

IndicTrans2 was trained on a collection called BPCC, and **BPCC already contains
Samanantar**. Marathi is also one of the 22 languages the model was built for
from the start. So I was training a model on material it had already seen, in a
language it already handled well.

I wasn't teaching it anything new. I was going over old ground.

You can see this in the loss curve. It drops quickly in the first 200 steps —
which is mostly just the learning rate warming up — and then barely moves for the
remaining 3,550 steps:

```
step   50:  4.18
step  200:  3.83
step 3750:  3.74
```

That flat line isn't a bug. There simply wasn't much left for the model to learn.

And the final scores confirm it. On an independent benchmark of 500 sentences,
the fine-tuned model came out **worse** than the one I started with:

| | chrF++ | BLEU |
| --- | ---: | ---: |
| Original model | 50.72 | 15.15 |
| After fine-tuning | 48.72 | 13.44 |

Two points worse. Not catastrophic, but clearly the wrong direction, and entirely
consistent with the story above: there was no headroom to gain, so the small
portion of the model I was allowed to change drifted toward the quirks of a
web-scraped corpus — including its misaligned pairs — and that costs accuracy on
clean text.

There is a second measurement worth putting next to that one. I also scored the
1,000 Samanantar pairs I had set aside before training and never used. The
original model scores **38.70** there, against **50.72** on the independent
benchmark — twelve points lower on data drawn from the very corpus I trained on.

That gap says something about the references rather than the model. A translation
can be perfectly good and still disagree with a reference that a web scraper
matched up incorrectly. It is a concrete measure of how noisy the training data
is, and it reinforces why filtering mattered.

**What I'd do instead:** use data the model hasn't seen. Bhili would be the
obvious choice — a genuinely low-resource language where the model has little or
no coverage. Fine-tuning there would be adding a capability rather than
rehearsing one. Failing that, a specific domain of Marathi the base model handles
poorly — legal or medical text, or conversational speech.

I kept the Samanantar run because it demonstrates the full pipeline working end
to end, and because the flat result is itself informative. But if the goal were a
better model rather than a working process, I'd change the data first.

---

## 3. Cleaning the data

Samanantar wasn't assembled by hand. A program crawled the web, found English and
Marathi pages that looked related, and guessed which sentences correspond. It
guesses wrong a fair amount.

So before training on anything I filtered it, and made the filters report what
they removed:

| Filter | What it catches | Removed |
| --- | --- | ---: |
| Blank rows | One side empty | 0 |
| Length bounds | Fragments and headers, or runaway rows | 5,166 |
| Length ratio | One side far longer than the other — usually misaligned | 2,036 |
| Script check | "Marathi" that's actually English, a URL, or a table of numbers | 106 |
| Copies | Source identical to target — the matcher gave up | 0 |
| Duplicates | The same pair repeated | 0 |
| Repeated sources | Same English sentence with several Marathi versions | 9,120 |

**121,000 pairs survived out of 137,428 — 88%.**

The script check is the one I'm happiest with. It counts what fraction of the
Marathi side is actually written in Devanagari script. Real Marathi scores above
90%. English scores zero. A URL scores zero. I set the bar at 55%, which lets
through a Marathi sentence containing an English brand name but rejects a row
that's mostly English.

Two things surprised me. There were **no** exact duplicate pairs, which suggests
Samanantar was already cleaned that way before release. And the largest single
filter was repeated English sources — the corpus really does carry several
different Marathi translations of the same English sentence.

I also deliberately used only 121,000 of the 3.63 million pairs. That's about 3%.
It's a budget decision, not a limitation — the code streams the data and stops
early, so the rest is never even downloaded, and one training run fits inside a
single free Colab session.

---

## 4. Why LoRA instead of normal fine-tuning

Normal fine-tuning updates every number in the model. This model has 218 million
of them. During training you need to hold the weights, a copy of the gradients,
and the optimiser's bookkeeping for each one — roughly three to four times the
model's size in GPU memory.

LoRA takes a different approach: freeze the original model entirely, and bolt on
small extra matrices that learn the adjustment. Here that meant **6.5 million
trainable numbers instead of 218 million — about 3%.**

Why I chose it:

- **Memory.** The optimiser only tracks the small added pieces, so there's plenty
  of room left for a decent batch size on a free Colab T4.
- **The saved result is tiny.** The output is a few megabytes rather than a
  gigabyte, which makes it easy to share.
- **It's harder to wreck the model.** The original weights don't move, so a bad
  learning rate damages the adapter rather than the pretrained model underneath.

The honest trade-off: LoRA can only adjust, not overhaul. Given that this
particular fine-tune had little to learn anyway, that wasn't the limiting factor.
Full fine-tuning would also have fitted in memory at this model size, and the
code supports switching with one flag.

---

## 5. The checks I ran before training

The expensive way to fail at this is to start a multi-hour run with a subtle bug,
watch a perfectly normal-looking loss curve, and discover at the end that the
model learned nothing. I built two checks to make that impossible.

**Check one: which vocabulary encodes the Marathi?**

This model keeps two separate vocabularies — one that understands English
letters, one that understands Devanagari. If you accidentally use the English one
to encode the Marathi training targets, every Marathi character becomes "unknown
symbol". Training still runs. The loss still falls smoothly. The model learns to
output gibberish, because gibberish is genuinely the best answer to the question
it was asked.

Nothing warns you. So I wrote a script that takes a Marathi sentence, converts it
to numbers, converts it back, and checks it survives the round trip. It also
measures how many pieces the sentence broke into: a healthy vocabulary uses well
under one piece per character, and anything near one means it's falling back to
splitting character by character — the signature of the wrong vocabulary.

**Check two: can it memorise 32 sentences?**

Before the real run, I train on just 32 pairs for 30 passes. A model this size
should be able to memorise 32 sentences almost perfectly. If the loss doesn't
collapse, something in the training setup is broken and I've found out in two
minutes instead of after an hour.

I got this check wrong the first time, in an instructive way. I set the pass mark
at "loss below 1.0" — but with the settings I was using, the loss mathematically
cannot go below **1.363**. I'd written a test that could never pass, and it
dutifully reported a working pipeline as broken.

The cause is a technique called label smoothing. Instead of training the model to
be 100% certain of each word, you train it toward about 90% certain, which stops
it becoming overconfident. The side effect is that even a perfect model carries
some leftover error — a floor the loss can't go below. I now calculate that floor
from the actual vocabulary size and set the threshold above it, and I turn
smoothing off entirely for the memorisation check so that zero is genuinely
reachable.

---

## 6. The problem that cost the most time: the KV cache

This was the single biggest time sink, and it's worth explaining properly because
the cause is subtle.

**What the cache does.** When the model writes a translation, it produces one
word at a time. At each step it looks back over everything it has written so far.
Without a cache, it recalculates all of that history from scratch at every single
step. With a cache, it remembers the previous work and only computes the new
word. For a 128-word output that's the difference between 128 units of work and
about 8,000.

**Why it didn't work.** IndicTrans2 doesn't ship as ordinary library code — it
comes with its own model file downloaded from the model hub. That file was
written for an older version of the `transformers` library. In the old version,
the library passed "nothing yet" as an empty value on the first step. In the
current version it passes an empty *container* instead. The model file checks "is
this nothing?", gets told "no, it's a container", tries to read from the empty
container, and crashes.

**What I tried.** First I assumed I'd picked a badly maintained community version
of the model, so I switched to the official AI4Bharat release. It has exactly the
same problem — the code hadn't been updated there either. That detour cost real
time and taught me to check assumptions with a measurement rather than a guess.

**How I resolved it.** I turned the cache off and, crucially, *measured* what
that actually costs rather than assuming. The answer was **0.19 seconds per
sentence**, meaning a 500-sentence evaluation takes about 95 seconds. Perfectly
acceptable.

And here's the thing I should have worked out sooner: **training never uses the
cache at all.** During training the model is shown the entire correct answer at
once and predicts all positions in a single pass — there's no word-by-word
writing happening. The cache only matters when generating. So the whole problem
only ever affected evaluation, not the 52-minute training run.

I wrote a small script that checks any model for this in about a minute and
reports the actual speed cost, so the decision is based on a number rather than a
worry.

---

## 7. The other things that broke

**The model wouldn't train at all, at first.** It kept insisting I hadn't given
it the decoder input. Most translation models will work this out for themselves —
you hand them the correct answer, and internally they shift it by one position to
create the input. Position one sees the start marker and must predict word one;
position two sees word one and must predict word two; and so on. This model
doesn't do that step, and doesn't expose the standard hook for the library to do
it either. So I do it myself in the batching code.

The clue that made it click: translation worked fine, only training failed. When
generating, the model builds its own decoder input, so that path was never
exercising the missing piece.

**A version conflict in the LoRA library.** PEFT checks whether an optional
compression library is installed, and when it finds an old version it raises an
error instead of just skipping it. Colab ships that old version by default, so
every LoRA run died before it started. Uninstalling the unused library fixes it,
and the code now turns that crash into a message saying exactly that.

**Evaluation during training was eating the run.** I originally had the model
translate 1,000 validation sentences every 500 steps. With the cache off, that's
about 50 minutes per evaluation against a 52-minute training run — seven of them
would have turned one hour into seven.

I turned it off, and on reflection it was measuring the wrong thing anyway. The
validation sentences come from the same web-scraped corpus as the training data,
so scoring well on them mostly proves you've learned that corpus's quirks. I
measure quality once, properly, at the end, on a benchmark the model has never
seen.

**A bug in my own logging** that cost more time than any of the above. The status
line reporting whether evaluation was on or off was checking for a lowercase "no"
in a value that was spelled uppercase. It confidently announced "evaluation is
ON" during runs where evaluation was correctly off. I spent a long time chasing a
problem that didn't exist, because my own diagnostic was lying to me.

The lesson I actually took from that: **a diagnostic you trust is part of the
system, and it needs checking like everything else.**

---

## 8. How I measure the result

I score against an independent benchmark — written by people, covering topics
unrelated to the training data, and never seen during training. The run above
used FLORES-200, 500 sentences from its `devtest` split.

Getting the benchmark to load was its own small lesson. My first choice was
IN22-Gen, AI4Bharat's own benchmark, but its dataset page describes one layout
and the actual published files use another — the page documents per-language
configurations and a split called `gen`, while what is actually there is a single
default configuration with a `test` split. Rather than hardcode one guess, the
loader tries a list of plausible shapes and reports which succeeded, so a
mismatch costs a warning line instead of a failed run at the end of an hour.

I deliberately do *not* use held-out sentences from Samanantar as the headline
number. Those come from the same scrape as the training data, so scoring well on
them partly reflects having learned that scrape's habits rather than having
learned better Marathi. I do score them as well, though, because the **gap
between the two numbers** is the interesting part: if the in-house number
improves while the independent one doesn't, that tells you exactly what the model
actually learned.

The main measure is **chrF++**, which compares translations at the level of
character sequences. This matters for Marathi, where words change form heavily
depending on their grammatical role. A word-matching measure like BLEU marks a
correct translation wrong if it inflects a word differently from the reference.
I report BLEU too, because people expect it, but I trust chrF++ more here.

One more decision: with evaluation switched off during training, nothing
automatically picks the best checkpoint. The tempting shortcut is to take the one
with the lowest training loss, but training loss measures how well the model fits
the sentences it just practised on, so choosing that way selects for
memorisation. I use the final checkpoint instead, which is the right call for a
single pass with a learning rate decaying to near zero — the schedule is designed
so the end state is the settled one. Over several passes, where later
checkpoints start overfitting, I would score each one against the independent
benchmark and choose on that.

---

## 9. What I'd change

**Different data, first and foremost.** Everything else here worked. The reason
the result is flat is that I trained on material the model already knew. Bhili,
or any Marathi corpus outside BPCC, would change that outcome more than any
amount of hyperparameter tuning.

**Better filtering.** My filters are structural — lengths, ratios, which script
the text is in. They can't tell whether two sentences actually mean the same
thing. A meaning-based similarity score would catch pairs that pass every check I
have while being unrelated.

**A fair comparison against full fine-tuning.** I chose LoRA on memory grounds
and never tested whether full fine-tuning would have done better at the same
number of steps.

**Read the output properly.** Automatic scores hide things — a translation can
lose a name, repeat itself, or stop halfway and still score reasonably. I look at
fifteen examples by hand; a proper review would be larger and would ask a Marathi
speaker.

**Break the benchmark down by topic** instead of reporting one overall number,
to see where the fine-tuning helped and where it hurt.

---

## 10. In short

The pipeline works. It streams and filters a large noisy corpus, verifies the
tricky parts before spending money on them, trains, and measures the result
against an independent benchmark. The checks caught real bugs — a missing decoder
input, an impossible test threshold, a lying status line.

The result is a model that's barely different from the one I started with, and I
can explain exactly why: I fine-tuned it on data it had already been trained on.
That's a data selection mistake, it's visible in the loss curve, and it's the
first thing I'd fix.
