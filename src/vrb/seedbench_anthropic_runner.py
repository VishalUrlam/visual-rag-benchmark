"""Run SEED-Bench MCQ samples through Anthropic Claude."""

from __future__ import annotations

import base64
from pathlib import Path

import anthropic

from .config import settings
from .seedbench_evaluator import SEEDBenchItemResult, SEEDBenchResult, _mcq_match
from .seedbench_loader import SEEDBenchSample

_SYSTEM = (
    "You are a precise visual question answering assistant. "
    "Answer only with the letter of the correct option (A, B, C, or D). No explanation."
)


def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def _build_prompt(question: str, options: dict[str, str]) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"{question}\n\n{opts}"


def run_seedbench_anthropic(
    samples: list[SEEDBenchSample],
    *,
    model: str = "claude-opus-4-7",
    log=print,
) -> SEEDBenchResult:
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
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
            response = client.messages.create(
                model=model,
                max_tokens=16,
                system=_SYSTEM,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt},
                ]}],
            )
            prediction = response.content[0].text.strip() if response.content else ""
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
