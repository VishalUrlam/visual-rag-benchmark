from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class FileType(str, Enum):
    PDF = "pdf"
    DOC = "doc"
    DOCX = "docx"
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"


class GroundTruth(BaseModel):
    query: str
    expected_facts: list[str] = Field(
        description="Substrings that MUST appear in retrieved content"
    )
    should_not_contain: list[str] = Field(
        default_factory=list,
        description="Substrings that must NOT appear — hallucination probes",
    )


class TestCase(BaseModel):
    name: str
    file_path: str
    file_type: FileType
    description: str = ""
    queries: list[GroundTruth]


class RetrievedChunk(BaseModel):
    content: str
    score: float = 0.0
    metadata: dict = Field(default_factory=dict)


class QueryEvaluation(BaseModel):
    query: str
    expected_facts: list[str]
    retrieved_chunks: list[RetrievedChunk]
    facts_found: list[str]
    facts_missing: list[str]
    hallucinations_detected: list[str]
    precision: float
    recall: float
    f1: float


class FileResult(BaseModel):
    test_case_name: str
    file_path: str
    file_type: str
    platform: str
    doc_id: str | None = None
    ingestion_success: bool
    ingestion_error: str | None = None
    query_evaluations: list[QueryEvaluation] = Field(default_factory=list)
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_f1: float = 0.0


class BenchmarkReport(BaseModel):
    platform: str
    run_at: datetime = Field(default_factory=datetime.utcnow)
    file_results: list[FileResult]
    overall_precision: float
    overall_recall: float
    overall_f1: float
    hallucination_rate: float
