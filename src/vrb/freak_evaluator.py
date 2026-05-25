"""Evaluator for FREAK benchmark predictions.

Usage
-----
MCQ (multiple-choice):

    from vrb.freak_loader import load_mcq
    from vrb.freak_evaluator import evaluate_mcq

    samples = load_mcq(Path("data/freak/dataset.json"))
    predictions = ["3.", "2.", ...]   # one string per sample, in order
    result = evaluate_mcq(samples, predictions)
    print(result.summary())

QA (free-form):

    from vrb.freak_loader import load_qa
    from vrb.freak_evaluator import evaluate_qa

    samples = load_qa(Path("data/freak/dataset_qa.json"))
    predictions = ["No plaque is visible.", ...]
    result = evaluate_qa(samples, predictions)
    print(result.summary())

Prediction format
-----------------
MCQ  — the full option text (e.g. "3.") **or** the letter (A/B/C/D).
       Both are matched against the ground_truth option text.
QA   — any free-form string. The evaluator checks whether the ground_truth
       is a substring of the prediction (case-insensitive). If `hallu_answer`
       is present and the prediction matches it instead, the item is counted
       as a hallucination.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .freak_loader import FREAKMCQSample, FREAKQASample

CATEGORIES = ["detection", "counting", "attribute", "analysis", "position", "ocr",
              "color", "shape", "logic"]


@dataclass
class MCQItemResult:
    id: int
    question: str
    ground_truth: str
    prediction: str
    correct: bool
    category: str | list[str]


@dataclass
class QAItemResult:
    id: int
    question: str
    ground_truth: str
    prediction: str
    correct: bool
    hallucinated: bool
    category: str | list[str]


@dataclass
class FREAKMCQResult:
    items: list[MCQItemResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.items:
            return 0.0
        return sum(r.correct for r in self.items) / len(self.items)

    def accuracy_by_category(self) -> dict[str, float]:
        cat_correct: dict[str, int] = {c: 0 for c in CATEGORIES}
        cat_total: dict[str, int] = {c: 0 for c in CATEGORIES}
        for r in self.items:
            cats = r.category if isinstance(r.category, list) else [r.category]
            for c in cats:
                if c in cat_correct:
                    cat_total[c] += 1
                    cat_correct[c] += int(r.correct)
        return {
            c: (cat_correct[c] / cat_total[c]) if cat_total[c] else float("nan")
            for c in CATEGORIES
        }

    def summary(self) -> str:
        lines = [f"FREAK MCQ — {len(self.items)} samples", f"  Overall accuracy: {self.accuracy:.1%}"]
        for cat, acc in self.accuracy_by_category().items():
            if acc == acc:  # skip NaN
                lines.append(f"  {cat:<12}: {acc:.1%}")
        return "\n".join(lines)


@dataclass
class FREAKQAResult:
    items: list[QAItemResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.items:
            return 0.0
        return sum(r.correct for r in self.items) / len(self.items)

    @property
    def hallucination_rate(self) -> float:
        if not self.items:
            return 0.0
        return sum(r.hallucinated for r in self.items) / len(self.items)

    def accuracy_by_category(self) -> dict[str, float]:
        cat_correct: dict[str, int] = {c: 0 for c in CATEGORIES}
        cat_total: dict[str, int] = {c: 0 for c in CATEGORIES}
        for r in self.items:
            cats = r.category if isinstance(r.category, list) else [r.category]
            for c in cats:
                if c in cat_correct:
                    cat_total[c] += 1
                    cat_correct[c] += int(r.correct)
        return {
            c: (cat_correct[c] / cat_total[c]) if cat_total[c] else float("nan")
            for c in CATEGORIES
        }

    def summary(self) -> str:
        lines = [
            f"FREAK QA — {len(self.items)} samples",
            f"  Overall accuracy:       {self.accuracy:.1%}",
            f"  Hallucination rate:     {self.hallucination_rate:.1%}",
        ]
        for cat, acc in self.accuracy_by_category().items():
            if acc == acc:
                lines.append(f"  {cat:<12}: {acc:.1%}")
        return "\n".join(lines)


def _normalize(text: str) -> str:
    return text.strip().lower().rstrip(".")


def _letter_to_option(letter: str, options: list[str]) -> str | None:
    """Convert A/B/C/D to the corresponding option text."""
    idx = ord(letter.upper()) - ord("A")
    if 0 <= idx < len(options):
        return options[idx]
    return None


def _mcq_match(prediction: str, ground_truth: str, options: list[str]) -> bool:
    pred = prediction.strip()
    # Direct text match
    if _normalize(pred) == _normalize(ground_truth):
        return True
    # Letter match (A/B/C/D)
    if re.fullmatch(r"[A-Da-d]\.?", pred):
        resolved = _letter_to_option(pred[0], options)
        if resolved and _normalize(resolved) == _normalize(ground_truth):
            return True
    return False


def evaluate_mcq(
    samples: list[FREAKMCQSample],
    predictions: list[str],
) -> FREAKMCQResult:
    if len(samples) != len(predictions):
        raise ValueError(
            f"samples ({len(samples)}) and predictions ({len(predictions)}) must have the same length"
        )
    result = FREAKMCQResult()
    for sample, pred in zip(samples, predictions):
        correct = _mcq_match(pred, sample.ground_truth, sample.options)
        result.items.append(
            MCQItemResult(
                id=sample.id,
                question=sample.question,
                ground_truth=sample.ground_truth,
                prediction=pred,
                correct=correct,
                category=sample.category,
            )
        )
    return result


def evaluate_qa(
    samples: list[FREAKQASample],
    predictions: list[str],
) -> FREAKQAResult:
    """Evaluate free-form QA predictions.

    Correctness: ground_truth is a substring of the prediction (case-insensitive).
    Hallucination: the prediction matches the known hallu_answer more closely
                   than the ground_truth (hallu_answer substring found but
                   ground_truth substring not found).
    """
    if len(samples) != len(predictions):
        raise ValueError(
            f"samples ({len(samples)}) and predictions ({len(predictions)}) must have the same length"
        )
    result = FREAKQAResult()
    for sample, pred in zip(samples, predictions):
        pred_lower = pred.lower()
        gt_lower = sample.ground_truth.lower()
        correct = gt_lower in pred_lower

        hallucinated = False
        if sample.hallu_answer:
            hallu_lower = sample.hallu_answer.lower()
            # Hallucination: hallu keywords present and ground truth absent
            hallucinated = (hallu_lower in pred_lower) and not correct

        result.items.append(
            QAItemResult(
                id=sample.id,
                question=sample.question,
                ground_truth=sample.ground_truth,
                prediction=pred,
                correct=correct,
                hallucinated=hallucinated,
                category=sample.category,
            )
        )
    return result
