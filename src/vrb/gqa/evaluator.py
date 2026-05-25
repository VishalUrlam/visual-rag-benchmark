"""Evaluate GQA predictions against ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field

from .loader import GQASample


@dataclass
class GQAItemResult:
    id: str
    image_id: str
    image_path: str
    question: str
    ground_truth: str
    prediction: str
    correct: bool
    category: str
    type: str


@dataclass
class GQAResult:
    items: list[GQAItemResult] = field(default_factory=list)

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


def _match(prediction: str, ground_truth: str) -> bool:
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()
    return pred == gt or gt in pred


def evaluate_gqa(samples: list[GQASample], predictions: list[str]) -> GQAResult:
    result = GQAResult()
    for sample, prediction in zip(samples, predictions):
        correct = _match(prediction, sample.answer)
        result.items.append(GQAItemResult(
            id=sample.id,
            image_id=sample.image_id,
            image_path=str(sample.image_path),
            question=sample.question,
            ground_truth=sample.answer,
            prediction=prediction,
            correct=correct,
            category=sample.category,
            type=sample.type,
        ))
    return result
