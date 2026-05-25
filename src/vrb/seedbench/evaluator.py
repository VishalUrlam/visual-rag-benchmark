"""Evaluate SEED-Bench MCQ predictions."""

from __future__ import annotations

from dataclasses import dataclass, field

from .loader import SEEDBenchSample


@dataclass
class SEEDBenchItemResult:
    id: str
    image_path: str
    question: str
    options: dict[str, str]
    ground_truth: str
    prediction: str
    correct: bool
    category: str
    level: str


@dataclass
class SEEDBenchResult:
    items: list[SEEDBenchItemResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.items:
            return 0.0
        return sum(1 for i in self.items if i.correct) / len(self.items)

    def accuracy_by_category(self) -> dict[str, float]:
        cats: dict[str, list[bool]] = {}
        for item in self.items:
            cats.setdefault(item.category, []).append(item.correct)
        return {c: sum(v) / len(v) for c, v in cats.items()}


def _mcq_match(prediction: str, answer: str) -> bool:
    pred = prediction.strip().upper()
    return pred == answer or pred.startswith(answer)
