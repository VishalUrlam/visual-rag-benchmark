"""Run GQA samples through Anthropic Claude vision models."""

from __future__ import annotations

import base64
from pathlib import Path

import anthropic

from .config import settings
from .gqa_evaluator import GQAItemResult, GQAResult, _match
from .gqa_loader import GQASample

_SYSTEM = (
    "You are a precise visual question answering assistant. "
    "Answer the question with a single word or very short phrase. No explanation."
)


def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def _ask(client: anthropic.Anthropic, model: str, image_path: Path, question: str) -> str:
    b64 = _encode_image(image_path)
    response = client.messages.create(
        model=model,
        max_tokens=64,
        system=_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": question},
            ],
        }],
    )
    return response.content[0].text.strip() if response.content else ""


def run_gqa_anthropic(
    samples: list[GQASample],
    *,
    model: str = "claude-opus-4-7",
    log=print,
) -> GQAResult:
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    result = GQAResult()

    for i, sample in enumerate(samples, 1):
        log(f"  {i}/{len(samples)} — id={sample.id}")
        if not sample.image_path.exists():
            log(f"    ERR  Image not found: {sample.image_path}")
            result.items.append(GQAItemResult(
                id=sample.id, image_id=sample.image_id,
                image_path=str(sample.image_path),
                question=sample.question, ground_truth=sample.answer,
                prediction="", correct=False,
                category=sample.category, type=sample.type,
            ))
            continue

        try:
            prediction = _ask(client, model, sample.image_path, sample.question)
        except Exception as exc:
            log(f"    ERR  {exc}")
            prediction = ""

        correct = _match(prediction, sample.answer)
        log(f"    {'✓' if correct else '✗'}  pred={prediction!r}  gt={sample.answer!r}")
        result.items.append(GQAItemResult(
            id=sample.id, image_id=sample.image_id,
            image_path=str(sample.image_path),
            question=sample.question, ground_truth=sample.answer,
            prediction=prediction, correct=correct,
            category=sample.category, type=sample.type,
        ))

    return result
