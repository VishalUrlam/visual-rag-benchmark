"""Run SEED-Bench MCQ samples through OpenRouter (Gemini models)."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from openai import AsyncOpenAI

from ..config import settings
from .evaluator import SEEDBenchItemResult, SEEDBenchResult, _mcq_match
from .loader import SEEDBenchSample

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

_SYSTEM = (
    "You are a precise visual question answering assistant. "
    "Answer only with the letter of the correct option (A, B, C, or D). No explanation."
)


def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def _build_prompt(question: str, options: dict[str, str]) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"{question}\n\n{opts}"


async def _run_all(samples: list[SEEDBenchSample], model: str, log) -> SEEDBenchResult:
    client = AsyncOpenAI(api_key=settings.openrouter_api_key, base_url=_OPENROUTER_BASE)
    result = SEEDBenchResult()

    for i, sample in enumerate(samples, 1):
        log(f"  {i}/{len(samples)} — id={sample.id} ({sample.category})")
        if not sample.image_path.exists():
            log(f"    ERR  Image not found: {sample.image_path}")
            result.items.append(SEEDBenchItemResult(
                id=sample.id, image_path=str(sample.image_path),
                question=sample.question, options=sample.options,
                ground_truth=sample.answer, prediction="",
                correct=False, category=sample.category, level=sample.level,
            ))
            continue

        prompt = _build_prompt(sample.question, sample.options)
        try:
            b64 = _encode_image(sample.image_path)
            response = await client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ]},
                ],
            )
            prediction = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            log(f"    ERR  {exc}")
            prediction = ""

        correct = _mcq_match(prediction, sample.answer)
        log(f"    {'✓' if correct else '✗'}  pred={prediction!r}  gt={sample.answer!r}")
        result.items.append(SEEDBenchItemResult(
            id=sample.id, image_path=str(sample.image_path),
            question=sample.question, options=sample.options,
            ground_truth=sample.answer, prediction=prediction,
            correct=correct, category=sample.category, level=sample.level,
        ))

    return result


def run_seedbench_openrouter(
    samples: list[SEEDBenchSample],
    *,
    model: str = "~google/gemini-pro-latest",
    log=print,
) -> SEEDBenchResult:
    return asyncio.run(_run_all(samples, model, log))
