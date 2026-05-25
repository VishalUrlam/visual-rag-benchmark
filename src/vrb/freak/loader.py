"""Loader for the FREAK hallucination benchmark dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FREAKMCQSample:
    id: int
    image_path: Path
    question: str
    item: str
    category: str | list[str]
    options: list[str]
    ground_truth: str


@dataclass
class FREAKQASample:
    id: int
    image_path: Path
    question: str
    item: str
    category: str | list[str]
    ground_truth: str
    hallu_answer: str = ""


def load_mcq(dataset_path: Path, images_dir: Path | None = None) -> list[FREAKMCQSample]:
    """Load dataset.json (multiple-choice questions)."""
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    img_dir = images_dir or dataset_path.parent / "images"
    return [
        FREAKMCQSample(
            id=item["id"],
            image_path=img_dir / item["image_path"],
            question=item["question"],
            item=item["item"],
            category=item["category"],
            options=item["options"],
            ground_truth=item["ground_truth"],
        )
        for item in raw
    ]


def load_qa(dataset_path: Path, images_dir: Path | None = None) -> list[FREAKQASample]:
    """Load dataset_qa.json (free-form questions)."""
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    img_dir = images_dir or dataset_path.parent / "images"
    return [
        FREAKQASample(
            id=item["id"],
            image_path=img_dir / item["image_path"],
            question=item["question"],
            item=item["item"],
            category=item["category"],
            ground_truth=item["ground_truth"],
            hallu_answer=item.get("hallu_answer", ""),
        )
        for item in raw
    ]
