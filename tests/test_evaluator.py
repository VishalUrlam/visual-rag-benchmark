from vrb.evaluator import evaluate_query
from vrb.models import GroundTruth, RetrievedChunk


def _chunks(*texts: str) -> list[RetrievedChunk]:
    return [RetrievedChunk(content=t, score=1.0) for t in texts]


def test_all_facts_found():
    gt = GroundTruth(query="q", expected_facts=["apple", "banana"])
    result = evaluate_query(gt, _chunks("I like apple and banana."))
    assert result.precision == 1.0
    assert result.f1 == 1.0
    assert result.facts_missing == []


def test_partial_facts_found():
    gt = GroundTruth(query="q", expected_facts=["apple", "banana"])
    result = evaluate_query(gt, _chunks("Only apple here."))
    assert result.precision == 0.5
    assert "banana" in result.facts_missing


def test_no_facts_found():
    gt = GroundTruth(query="q", expected_facts=["apple"])
    result = evaluate_query(gt, _chunks("Nothing relevant."))
    assert result.precision == 0.0
    assert result.f1 == 0.0


def test_hallucination_detected():
    gt = GroundTruth(
        query="q",
        expected_facts=["apple"],
        should_not_contain=["unicorn"],
    )
    result = evaluate_query(gt, _chunks("apple and unicorn appear here"))
    assert "unicorn" in result.hallucinations_detected


def test_no_hallucination():
    gt = GroundTruth(
        query="q",
        expected_facts=["apple"],
        should_not_contain=["unicorn"],
    )
    result = evaluate_query(gt, _chunks("just apple here"))
    assert result.hallucinations_detected == []


def test_empty_chunks():
    gt = GroundTruth(query="q", expected_facts=["apple"])
    result = evaluate_query(gt, [])
    assert result.precision == 0.0
