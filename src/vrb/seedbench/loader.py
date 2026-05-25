"""Load SEED-Bench sample dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SEEDBenchSample:
    id: str
    image_path: Path
    question: str
    options: dict[str, str]  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    answer: str
    category: str
    level: str


def load_seedbench(dataset_path: Path, images_dir: Path | None = None) -> list[SEEDBenchSample]:
    with open(dataset_path) as f:
        data = json.load(f)

    base = images_dir or dataset_path.parent / "images"
    return [
        SEEDBenchSample(
            id=item["id"],
            image_path=base / item["image_file"],
            question=item["question"],
            options=item["options"],
            answer=item["answer"],
            category=item.get("category", ""),
            level=item.get("level", ""),
        )
        for item in data
    ]
