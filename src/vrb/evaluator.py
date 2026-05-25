from .models import GroundTruth, QueryEvaluation, RetrievedChunk


def evaluate_query(
    ground_truth: GroundTruth,
    chunks: list[RetrievedChunk],
) -> QueryEvaluation:
    combined = " ".join(c.content for c in chunks).lower()

    facts_found = [f for f in ground_truth.expected_facts if f.lower() in combined]
    facts_missing = [f for f in ground_truth.expected_facts if f.lower() not in combined]
    hallucinations = [f for f in ground_truth.should_not_contain if f.lower() in combined]

    total = len(ground_truth.expected_facts)
    precision = len(facts_found) / total if total else 0.0
    recall = precision  # single-label set: precision == recall here
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return QueryEvaluation(
        query=ground_truth.query,
        expected_facts=ground_truth.expected_facts,
        retrieved_chunks=chunks,
        facts_found=facts_found,
        facts_missing=facts_missing,
        hallucinations_detected=hallucinations,
        precision=precision,
        recall=recall,
        f1=f1,
    )
