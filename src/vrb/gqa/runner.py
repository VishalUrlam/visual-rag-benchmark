"""Run GQA samples through OpenRouter (Gemini models)."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from openai import AsyncOpenAI

from ..config import settings
from .evaluator import GQAItemResult, GQAResult, _match
from .loader import GQASample

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

_SYSTEM = (
    "You are a precise visual question answering assistant. "
    "Answer the question with a single word or very short phrase. No explanation."
)


def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


async def _run_all(samples: list[GQASample], model: str, log) -> GQAResult:
    client = AsyncOpenAI(api_key=settings.openrouter_api_key, base_url=_OPENROUTER_BASE)
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
            b64 = _encode_image(sample.image_path)
            response = await client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": sample.question},
                    ]},
                ],
            )
            prediction = (response.choices[0].message.content or "").strip()
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


def run_gqa_openrouter(
    samples: list[GQASample],
    *,
    model: str = "~google/gemini-pro-latest",
    log=print,
) -> GQAResult:
    return asyncio.run(_run_all(samples, model, log))
