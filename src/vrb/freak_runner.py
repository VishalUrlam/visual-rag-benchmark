"""Run FREAK benchmark samples through a RAG client.

For each sample the runner:
  1. Ingests the image into the platform.
  2. Waits briefly for indexing.
  3. Searches with the visual question.
  4. Checks whether the ground-truth answer appears in the retrieved chunks.
  5. Optionally checks for known hallucination strings (QA only).
  6. Cleans up the document.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

from .clients.base import BaseRAGClient
from .freak_loader import FREAKMCQSample, FREAKQASample


@dataclass
class FREAKSampleResult:
    id: int
    question: str
    ground_truth: str
    category: str | list[str]
    retrieved_text: str
    correct: bool
    hallucinated: bool = False
    error: str = ""


@dataclass
class FREAKRunReport:
    platform: str
    mcq: list[FREAKSampleResult] = field(default_factory=list)
    qa: list[FREAKSampleResult] = field(default_factory=list)

    @property
    def mcq_accuracy(self) -> float:
        ok = [r for r in self.mcq if not r.error]
        return sum(r.correct for r in ok) / len(ok) if ok else 0.0

    @property
    def qa_accuracy(self) -> float:
        ok = [r for r in self.qa if not r.error]
        return sum(r.correct for r in ok) / len(ok) if ok else 0.0

    @property
    def qa_hallucination_rate(self) -> float:
        ok = [r for r in self.qa if not r.error]
        return sum(r.hallucinated for r in ok) / len(ok) if ok else 0.0

    def accuracy_by_category(self, results: list[FREAKSampleResult]) -> dict[str, float]:
        from collections import defaultdict
        correct: dict[str, int] = defaultdict(int)
        total: dict[str, int] = defaultdict(int)
        for r in results:
            if r.error:
                continue
            cats = r.category if isinstance(r.category, list) else [r.category]
            for c in cats:
                total[c] += 1
                correct[c] += int(r.correct)
        return {c: correct[c] / total[c] for c in total}


def _in_chunks(text: str, needle: str) -> bool:
    return needle.lower().strip() in text.lower()


async def _run_sample(
    client: BaseRAGClient,
    image_path: object,
    question: str,
    ground_truth: str,
    run_tag: str,
    hallu_answer: str = "",
    top_k: int = 5,
    cleanup: bool = True,
) -> tuple[str, bool, bool, str]:
    """Returns (retrieved_text, correct, hallucinated, error)."""
    from pathlib import Path
    image_path = Path(image_path)

    if not image_path.exists():
        return "", False, False, f"Image not found: {image_path}"

    doc_id = f"freak-{uuid.uuid4().hex[:8]}"
    try:
        doc_id = await client.ingest_file(
            image_path, doc_id, metadata={"tags": [run_tag]}
        )
    except Exception as exc:
        return "", False, False, f"Ingest failed: {exc}"

    # Poll until the document is indexed (up to 30s)
    indexed = False
    for _ in range(6):
        await asyncio.sleep(5)
        try:
            chunks = await client.search(question, top_k=top_k, tag=run_tag)
            if chunks:
                indexed = True
                break
        except Exception:
            pass

    if not indexed:
        if cleanup:
            try:
                await client.delete_document(doc_id)
            except Exception:
                pass
        return "", False, False, "Search returned no results after 30s"

    try:
        chunks = await client.search(question, top_k=top_k, tag=run_tag)
    except Exception as exc:
        chunks = []

    if cleanup:
        try:
            await client.delete_document(doc_id)
        except Exception:
            pass

    retrieved = " ".join(c.content for c in chunks)
    correct = _in_chunks(retrieved, ground_truth)
    hallucinated = bool(hallu_answer) and _in_chunks(retrieved, hallu_answer) and not correct
    return retrieved, correct, hallucinated, ""


async def run_freak_mcq(
    client: BaseRAGClient,
    samples: list[FREAKMCQSample],
    run_tag: str,
    *,
    top_k: int = 5,
    cleanup: bool = True,
    log=print,
) -> list[FREAKSampleResult]:
    results = []
    for i, sample in enumerate(samples, 1):
        log(f"  MCQ {i}/{len(samples)} — id={sample.id} ({sample.category})")
        retrieved, correct, _, error = await _run_sample(
            client,
            sample.image_path,
            sample.question,
            sample.ground_truth,
            run_tag=run_tag,
            top_k=top_k,
            cleanup=cleanup,
        )
        status = "✓" if correct else ("ERR" if error else "✗")
        log(f"    {status}  {error or ('correct' if correct else 'wrong')}")
        results.append(FREAKSampleResult(
            id=sample.id,
            question=sample.question,
            ground_truth=sample.ground_truth,
            category=sample.category,
            retrieved_text=retrieved,
            correct=correct,
            error=error,
        ))
    return results


async def run_freak_qa(
    client: BaseRAGClient,
    samples: list[FREAKQASample],
    run_tag: str,
    *,
    top_k: int = 5,
    cleanup: bool = True,
    log=print,
) -> list[FREAKSampleResult]:
    results = []
    for i, sample in enumerate(samples, 1):
        log(f"  QA  {i}/{len(samples)} — id={sample.id} ({sample.category})")
        retrieved, correct, hallucinated, error = await _run_sample(
            client,
            sample.image_path,
            sample.question,
            sample.ground_truth,
            run_tag=run_tag,
            hallu_answer=sample.hallu_answer,
            top_k=top_k,
            cleanup=cleanup,
        )
        status = "✓" if correct else ("ERR" if error else ("HALLU" if hallucinated else "✗"))
        log(f"    {status}  {error or ('correct' if correct else 'wrong')}")
        results.append(FREAKSampleResult(
            id=sample.id,
            question=sample.question,
            ground_truth=sample.ground_truth,
            category=sample.category,
            retrieved_text=retrieved,
            correct=correct,
            hallucinated=hallucinated,
            error=error,
        ))
    return results


async def run_freak_benchmark(
    client: BaseRAGClient,
    mcq_samples: list[FREAKMCQSample],
    qa_samples: list[FREAKQASample],
    *,
    top_k: int = 5,
    cleanup: bool = True,
    log=print,
) -> FREAKRunReport:
    import time
    run_tag = f"freak-{int(time.time())}"
    report = FREAKRunReport(platform=client.name)
    if mcq_samples:
        report.mcq = await run_freak_mcq(
            client, mcq_samples, run_tag, top_k=top_k, cleanup=cleanup, log=log
        )
    if qa_samples:
        report.qa = await run_freak_qa(
            client, qa_samples, run_tag, top_k=top_k, cleanup=cleanup, log=log
        )
    return report
