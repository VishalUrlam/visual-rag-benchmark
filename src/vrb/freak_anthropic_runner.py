"""Run FREAK benchmark samples directly through Anthropic Claude vision models."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import anthropic

from .config import settings
from .freak_evaluator import FREAKMCQResult, FREAKQAResult, MCQItemResult, QAItemResult, _mcq_match
from .freak_loader import FREAKMCQSample, FREAKQASample

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


def _ask(client: anthropic.Anthropic, model: str, system: str, image_path: Path, prompt: str) -> str:
    b64 = _encode_image(image_path)
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=system,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return response.content[0].text if response.content else ""


def run_mcq_anthropic(
    samples: list[FREAKMCQSample],
    *,
    model: str = "claude-opus-4-7",
    log=print,
) -> FREAKMCQResult:
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
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
            prediction = _ask(client, model, _MCQ_SYSTEM, sample.image_path, prompt)
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


def run_qa_anthropic(
    samples: list[FREAKQASample],
    *,
    model: str = "claude-opus-4-7",
    log=print,
) -> FREAKQAResult:
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
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
            prediction = _ask(client, model, _QA_SYSTEM, sample.image_path, sample.question)
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


def run_freak_anthropic(
    mcq_samples: list[FREAKMCQSample],
    qa_samples: list[FREAKQASample],
    *,
    model: str = "claude-opus-4-7",
    log=print,
) -> tuple[FREAKMCQResult | None, FREAKQAResult | None]:
    mcq_result = run_mcq_anthropic(mcq_samples, model=model, log=log) if mcq_samples else None
    qa_result = run_qa_anthropic(qa_samples, model=model, log=log) if qa_samples else None
    return mcq_result, qa_result
