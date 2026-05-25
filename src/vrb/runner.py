import asyncio
import uuid
from pathlib import Path

import yaml

from .clients.base import BaseRAGClient
from .evaluator import evaluate_query
from .models import BenchmarkReport, FileResult, TestCase


async def run_test_case(
    client: BaseRAGClient,
    test_case: TestCase,
    *,
    top_k: int = 5,
    cleanup: bool = True,
) -> FileResult:
    file_path = Path(test_case.file_path)
    doc_id = f"vrb-{uuid.uuid4().hex[:8]}"

    result = FileResult(
        test_case_name=test_case.name,
        file_path=str(file_path),
        file_type=test_case.file_type.value,
        platform=client.name,
        ingestion_success=False,
    )

    if not file_path.exists():
        result.ingestion_error = f"File not found: {file_path}"
        return result

    try:
        result.doc_id = await client.ingest_file(file_path, doc_id)
        result.ingestion_success = True
    except Exception as exc:
        result.ingestion_error = str(exc)
        return result

    # Brief pause to allow indexing
    await asyncio.sleep(2)

    evals = []
    for gt in test_case.queries:
        chunks = await client.search(gt.query, top_k=top_k)
        evals.append(evaluate_query(gt, chunks))

    result.query_evaluations = evals

    if evals:
        result.avg_precision = sum(e.precision for e in evals) / len(evals)
        result.avg_recall = sum(e.recall for e in evals) / len(evals)
        result.avg_f1 = sum(e.f1 for e in evals) / len(evals)

    if cleanup and result.doc_id:
        try:
            await client.delete_document(result.doc_id)
        except Exception:
            pass  # best-effort cleanup

    return result


def _aggregate(file_results: list[FileResult]) -> tuple[float, float, float, float]:
    successful = [r for r in file_results if r.ingestion_success and r.query_evaluations]
    if not successful:
        return 0.0, 0.0, 0.0, 0.0

    overall_precision = sum(r.avg_precision for r in successful) / len(successful)
    overall_recall = sum(r.avg_recall for r in successful) / len(successful)
    overall_f1 = sum(r.avg_f1 for r in successful) / len(successful)

    all_evals = [e for r in successful for e in r.query_evaluations]
    hallucination_rate = (
        sum(1 for e in all_evals if e.hallucinations_detected) / len(all_evals)
        if all_evals
        else 0.0
    )
    return overall_precision, overall_recall, overall_f1, hallucination_rate


async def run_benchmark(
    client: BaseRAGClient,
    test_cases: list[TestCase],
    *,
    top_k: int = 5,
    cleanup: bool = True,
) -> BenchmarkReport:
    file_results = []
    for tc in test_cases:
        result = await run_test_case(client, tc, top_k=top_k, cleanup=cleanup)
        file_results.append(result)

    precision, recall, f1, hallucination_rate = _aggregate(file_results)

    return BenchmarkReport(
        platform=client.name,
        file_results=file_results,
        overall_precision=precision,
        overall_recall=recall,
        overall_f1=f1,
        hallucination_rate=hallucination_rate,
    )


def load_test_cases(directory: Path) -> list[TestCase]:
    cases: list[TestCase] = []
    for yaml_file in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(yaml_file.read_text())
        if isinstance(raw, list):
            cases.extend(TestCase.model_validate(item) for item in raw)
        else:
            cases.append(TestCase.model_validate(raw))
    return cases
