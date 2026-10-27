# Reproducing the analyses

This document says which of the paper's results can be recomputed from this repository, and how.

## What can be recomputed

The [generation inventory](../data/generations/README.md) lists every released archive, its
manifest checks and the commands that recompute results from it. None of them needs a GPU.

- **The role result (E36) and the copy control (E40).** `affective_contagion.verification`
  recomputes the role table from the released readings. `copy_control.copy_control_by_inbox`
  recomputes the copy control: it rebuilds the messages each reader actually received and removes
  the same word set from both replies of a pair before scoring. The second scoring instrument (E43) rescores the same
  readings with `affective_contagion.emotion_probe` and the pinned classifier checkpoint.
- **The model-family, filter, closing-sentence, urgency and paraphrase results** (E39, E41, E44,
  E45, E49, E50). Their archives hold every reading and the inbox each reader saw, and
  `verification.verify_release` checks them against their manifests. No single command recomputes these tables.
- **The decision results (E46, E47, E48).** `scripts/analyze_reader_outcomes.py` computes paired
  Student t intervals and tests, with Holm correction across roles. Reported size is the numeric
  field in every reply, including HOLD; it is not an executed order or a filled position.
  `affective_contagion.reader_verification` computes the same means, intervals and Holm decisions
  by a separate route.
- **The injected design (E51).** `affective_contagion.injected_verification` rebuilds each pair from
  the archived run manifests, reads the sender from each manifest, checks the pairing gate, and
  derives the teammate response and the trade and price counts. Prices match across arms on three
  events; on the BUY event some differ, so the pooled teammate response there can include a
  response to price as well as to wording.

- **Two sender effects from the generated design.** The complete team runs were not exported, but
  the sender's shock-round messages survive in two inbox archives, because those runs supplied the
  later readers' stimuli. Rescoring them reproduces the deposit-outflow sender effect (E15,
  +0.0366 over 28 pairs, from the E36 inboxes, seeds 6–35) and the capacity-award sender effect
  (E31, +0.0252 over 24 pairs, from the E47 inboxes, with decision words scrubbed):

```python
import gzip, json, math, statistics
from pathlib import Path
from scipy import stats
from affective_contagion.scrub import scrub_decision_vocabulary
from affective_contagion.text_affect import VADLexicon

lexicon = VADLexicon.from_tsv(
    Path("data/raw/NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt"),
    scale="signed", match="surface_then_suffix",
)

def sender_effect(inboxes, seeds=None, scrub=False):
    """Paired alarmed-minus-calm arousal of the sender's message, with a 95% t interval."""
    texts = {}
    for row in map(json.loads, gzip.open(inboxes, "rt")):
        if seeds is None or row["seed"] in seeds:
            sender = next(m for m in row["messages"] if m["role"] == "sentiment_analyst")
            text = sender["decision_message"]
            texts.setdefault(row["seed"], {})[row["arm"]] = (
                scrub_decision_vocabulary(text) if scrub else text
            )
    diffs = [
        lexicon.score(arms["anxiety_like"]).arousal - lexicon.score(arms["neutral"]).arousal
        for arms in texts.values()
    ]
    mean, se = statistics.fmean(diffs), statistics.stdev(diffs) / math.sqrt(len(diffs))
    half = stats.t.ppf(0.975, len(diffs) - 1) * se
    return len(diffs), round(mean, 4), round(float(mean - half), 4), round(float(mean + half), 4)

root = Path("data/generations")
print("E15", sender_effect(root / "e36-role-context-inboxes.jsonl.gz", seeds=set(range(6, 36))))
print("E31 capacity award", sender_effect(root / "e47-action-buy-inboxes.jsonl.gz", scrub=True))
```

## What cannot

The rest of the generated-design runs were not exported: the teammate results (E18, and E31's
teammates), E31's covenant-note sender effect, the belief result (E30), the per-role breakdowns of
those runs, and the exploratory model-scale sweep (E19). Their configurations and code are here,
so the runs can be repeated, but those published numbers cannot be recomputed from released data.
GPU generation is not always reproducible, so a repeated run can differ in wording.

## Separately obtained inputs

The NRC VAD lexicon and the model weights are obtained under their own terms; see the repository
README. Rescoring with the emotion classifier needs the pinned classifier checkpoint. Classifier
scores do not show that agents experience emotions, and no human rating study was run.
