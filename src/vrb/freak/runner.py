"""Run FREAK benchmark samples through OpenRouter (any vision model)."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from openai import AsyncOpenAI

from ..config import settings
from .evaluator import FREAKMCQResult, FREAKQAResult, MCQItemResult, QAItemResult, _mcq_match
from .loader import FREAKMCQSample, FREAKQASample

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

_MCQ_SYSTEM = (
    "You are a precise visual question answering assistant. "
    "Answer only with the letter of the correct option (A, B, C, or D). No explanation."
)

_QA_SYSTEM = (
    "You are a precise visual question answering assistant. "
    "Answer the question concisely in one or two sentences. No explanation."
)


def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def _build_mcq_prompt(question: str, options: list[str]) -> str:
    letters = "ABCD"
    opts = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
    return f"{question}\n\n{opts}"


async def _ask(
    client: AsyncOpenAI,
    model: str,
    system: str,
    image_path: Path,
    prompt: str,
) -> str:
    b64 = _encode_image(image_path)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


async def _run_mcq(
    samples: list[FREAKMCQSample],
    *,
    model: str,
    log,
) -> FREAKMCQResult:
    client = AsyncOpenAI(api_key=settings.openrouter_api_key, base_url=_OPENROUTER_BASE)
    result = FREAKMCQResult()

    for i, sample in enumerate(samples, 1):
        log(f"  MCQ {i}/{len(samples)} — id={sample.id} ({sample.category})")
        if not sample.image_path.exists():
            log(f"    ERR  Image not found: {sample.image_path}")
            result.items.append(MCQItemResult(
                id=sample.id, question=sample.question,
                ground_truth=sample.ground_truth, prediction="",
                correct=False, category=sample.category,
            ))
            continue

        prompt = _build_mcq_prompt(sample.question, sample.options)
        try:
            prediction = await _ask(client, model, _MCQ_SYSTEM, sample.image_path, prompt)
        except Exception as exc:
            log(f"    ERR  {exc}")
            prediction = ""

        correct = _mcq_match(prediction, sample.ground_truth, sample.options)
        log(f"    {'✓' if correct else '✗'}  pred={prediction!r}  gt={sample.ground_truth!r}")
        result.items.append(MCQItemResult(
            id=sample.id, question=sample.question,
            ground_truth=sample.ground_truth, prediction=prediction,
            correct=correct, category=sample.category,
        ))

    return result


async def _run_qa(
    samples: list[FREAKQASample],
    *,
    model: str,
    log,
) -> FREAKQAResult:
    client = AsyncOpenAI(api_key=settings.openrouter_api_key, base_url=_OPENROUTER_BASE)
    result = FREAKQAResult()

    for i, sample in enumerate(samples, 1):
        log(f"  QA  {i}/{len(samples)} — id={sample.id} ({sample.category})")
        if not sample.image_path.exists():
            log(f"    ERR  Image not found: {sample.image_path}")
            result.items.append(QAItemResult(
                id=sample.id, question=sample.question,
                ground_truth=sample.ground_truth, prediction="",
                correct=False, hallucinated=False, category=sample.category,
            ))
            continue

        try:
            prediction = await _ask(client, model, _QA_SYSTEM, sample.image_path, sample.question)
        except Exception as exc:
            log(f"    ERR  {exc}")
            prediction = ""

        pred_lower = prediction.lower()
        gt_lower = sample.ground_truth.lower()
        correct = gt_lower in pred_lower
        hallucinated = (
            bool(sample.hallu_answer)
            and sample.hallu_answer.lower() in pred_lower
            and not correct
        )
        status = "✓" if correct else ("HALLU" if hallucinated else "✗")
        log(f"    {status}  pred={prediction!r}  gt={sample.ground_truth!r}")
        result.items.append(QAItemResult(
            id=sample.id, question=sample.question,
            ground_truth=sample.ground_truth, prediction=prediction,
            correct=correct, hallucinated=hallucinated, category=sample.category,
        ))

    return result


def run_freak_openrouter(
    mcq_samples: list[FREAKMCQSample],
    qa_samples: list[FREAKQASample],
    *,
    model: str,
    log=print,
) -> tuple[FREAKMCQResult | None, FREAKQAResult | None]:
    async def _run() -> tuple[FREAKMCQResult | None, FREAKQAResult | None]:
        mcq = await _run_mcq(mcq_samples, model=model, log=log) if mcq_samples else None
        qa = await _run_qa(qa_samples, model=model, log=log) if qa_samples else None
        return mcq, qa

    return asyncio.run(_run())
