"""Score text for named emotions with a human-annotated classifier, not a word list.

This is a second instrument beside `text_affect`. The NRC VAD lexicon used elsewhere scores
*anxious* and *urgent* almost identically. GoEmotions has separate `fear` and `nervousness` labels
and no urgency label, so it can indicate whether a reply expresses apprehension rather than only
sounding activated.

The classifier was trained on 58k Reddit comments labelled by people. It does not replace human
ratings of these texts. Its training domain is social media, not financial reporting, and that
shift is unmeasured. Its output is a classifier's verdict, not a human judgement.

Inference stays behind `EmotionScorer` so analysis, tests and the rest of the package run without
downloading anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# Pinned, because a classifier that silently changes revision changes every number it produced.
EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"
EMOTION_REVISION = "d75048347613a25d77de8cf6412eaae9fa7b26be"

# The apprehension labels. Both are kept rather than fear alone: GoEmotions splits an acute sense
# (fear) from a sustained one (nervousness), and anxiety-like language sits across both.
ANXIETY_LABELS = ("fear", "nervousness")
# Activated but not apprehensive. Reported beside the anxiety score so a rise in apprehension can
# be told apart from a rise in arousal generally.
AROUSED_NOT_ANXIOUS_LABELS = ("excitement", "surprise", "anger", "annoyance")


@dataclass(frozen=True, slots=True)
class EmotionScore:
    """Per-label probabilities for one text, plus the two summaries the discriminant needs."""

    labels: tuple[tuple[str, float], ...]

    def probability(self, label: str) -> float:
        """Return one label's probability, or zero when the model does not emit it."""
        return dict(self.labels).get(label, 0.0)

    @property
    def anxiety(self) -> float:
        """Summed probability of the apprehension labels."""
        return sum(self.probability(name) for name in ANXIETY_LABELS)

    @property
    def aroused_not_anxious(self) -> float:
        """Summed probability of labels that are activated without being apprehensive."""
        return sum(self.probability(name) for name in AROUSED_NOT_ANXIOUS_LABELS)

    @property
    def discriminant(self) -> float:
        """Anxiety minus activation-without-apprehension.

        The difference of two summed label probabilities. It does not identify anxiety uniquely:
        urgent wording also raises it, as the affect-by-urgency comparison (E44) shows.
        """
        return self.anxiety - self.aroused_not_anxious


@runtime_checkable
class EmotionScorer(Protocol):
    """Anything that can turn texts into per-label probabilities."""

    def score_all(self, texts: Sequence[str]) -> tuple[EmotionScore, ...]:
        """Score every text, in order, returning one result per input."""
        ...


@dataclass(slots=True)
class TransformersEmotionScorer:
    """GoEmotions classifier, loaded once and run in batches.

    Defaults to CPU. A roberta-base pass over a few thousand short messages takes minutes on CPU,
    which leaves the accelerator free for generation.
    """

    device: str = "cpu"
    batch_size: int = 64
    # `Any` rather than a transformers type: importing one here would make the whole package
    # depend on transformers, which is the coupling `EmotionScorer` exists to avoid.
    _model: Any = None
    _tokenizer: Any = None
    _labels: tuple[str, ...] = ()

    def _load(self) -> None:
        """Load the pinned model and tokenizer on first use."""
        if self._model is not None:
            return
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL, revision=EMOTION_REVISION)
        model = AutoModelForSequenceClassification.from_pretrained(
            EMOTION_MODEL, revision=EMOTION_REVISION
        )
        model.eval()
        model.to(self.device)
        self._model = model
        self._labels = tuple(model.config.id2label[i] for i in range(model.config.num_labels))

    def score_all(self, texts: Sequence[str]) -> tuple[EmotionScore, ...]:
        """Score every text. Multi-label, so probabilities are sigmoids and do not sum to one."""
        self._load()
        import torch  # noqa: PLC0415

        out: list[EmotionScore] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [text or "" for text in texts[start : start + self.batch_size]]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**encoded).logits
            out.extend(
                EmotionScore(labels=tuple(zip(self._labels, map(float, row), strict=True)))
                for row in torch.sigmoid(logits).cpu().tolist()
            )
        return tuple(out)
