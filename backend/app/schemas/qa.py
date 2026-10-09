"""FiniRAG QA API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QAAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str | None = None
    document_name: str | None = None
    document_type: str | None = None
    reporting_period: str | None = None
    company: str | None = None


class QAAskResponse(BaseModel):
    answer: str
    status: str
    answer_provider: str = "gemini"
    citations: list[dict] = Field(default_factory=list)
    financial_facts: list[dict] = Field(default_factory=list)
    calculations: list[dict] = Field(default_factory=list)
    validation_checks: list[dict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    retrieval_metadata: dict = Field(default_factory=dict)
    request_id: str = ""
    intents: list[str] = Field(default_factory=list)


class IndexStatusResponse(BaseModel):
    document_id: str
    status: str
    job_id: str | None = None
    chunk_count: int = 0
    index_version: str = "v1"
    attempt_count: int = 0
    error_code: str | None = None
    error_message: str | None = None
    updated_at: str | None = None


class ReindexResponse(BaseModel):
    document_id: str
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    document_id: str
    status: str
    attempt_count: int = 0
    index_version: str = "v1"
    error_code: str | None = None
    error_message: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
