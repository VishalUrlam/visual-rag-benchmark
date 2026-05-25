"""Load GQA sample dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GQASample:
    id: str
    image_id: str
    image_path: Path
    question: str
    answer: str
    full_answer: str
    category: str
    type: str


def load_gqa(dataset_path: Path, images_dir: Path | None = None) -> list[GQASample]:
    with open(dataset_path) as f:
        data = json.load(f)

    base = images_dir or dataset_path.parent / "images"
    samples = []
    for item in data:
        samples.append(GQASample(
            id=item["id"],
            image_id=item["image_id"],
            image_path=base / item["image_file"],
            question=item["question"],
            answer=item["answer"],
            full_answer=item["full_answer"],
            category=item.get("category", ""),
            type=item.get("type", ""),
        ))
    return samples
