"""Run GQA samples through OpenAI vision models."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from openai import AsyncOpenAI

from .config import settings
from .gqa_evaluator import GQAItemResult, GQAResult, _match
from .gqa_loader import GQASample

_SYSTEM = (
    "You are a precise visual question answering assistant. "
    "Answer the question with a single word or very short phrase. No explanation."
)


def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


async def _ask(client: AsyncOpenAI, model: str, image_path: Path, question: str) -> str:
    b64 = _encode_image(image_path)
    response = await client.chat.completions.create(
        model=model,
        max_completion_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": question},
            ]},
        ],
    )
    return response.choices[0].message.content.strip() if response.choices else ""


async def _run_all(
    samples: list[GQASample],
    model: str,
    log,
) -> GQAResult:
    client = AsyncOpenAI(api_key=settings.openai_api_key)
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
            prediction = await _ask(client, model, sample.image_path, sample.question)
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


def run_gqa_openai(
    samples: list[GQASample],
    *,
    model: str = "gpt-5.5",
    log=print,
) -> GQAResult:
    return asyncio.run(_run_all(samples, model, log))
