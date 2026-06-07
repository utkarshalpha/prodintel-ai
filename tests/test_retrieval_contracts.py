"""Phase 6D-i -- retrieval contract validation.

Covers validation rules, schema_version pinning, top_k / min_score bounds,
immutability, the meta count-partition invariant, and JSON serialization round-trip.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ai_contracts.enums import FrameworkName, KnowledgeSourceType
from app.ai_contracts.retrieval import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    RetrievalCitation,
    RetrievalMeta,
    RetrievalQuery,
    RetrievalResult,
)


def _citation(**overrides) -> RetrievalCitation:
    fields = dict(
        chunk_id=uuid4(),
        source_id=uuid4(),
        score=0.92,
        content="Reach x Impact x Confidence / Effort.",
        ordinal=0,
        corpus_version="pm-corpus-v1",
        source_title="RICE Reference",
        source_type="framework",
    )
    fields.update(overrides)
    return RetrievalCitation(**fields)


def _meta(**overrides) -> dict:
    fields = dict(
        embedding_model_id="hash-fake-v1",
        top_k=5,
        vector_match_count=1,
        resolved_count=1,
        dropped_stale_count=0,
        dropped_score_count=0,
    )
    fields.update(overrides)
    return fields


def _result(citations=None, meta=None, **overrides) -> RetrievalResult:
    fields = dict(
        query_text="how should I prioritize features?",
        corpus_version="pm-corpus-v1",
        citations=[_citation()] if citations is None else citations,
        meta=RetrievalMeta(**(meta if meta is not None else _meta())),
    )
    fields.update(overrides)
    return RetrievalResult(**fields)


# --- validation rules ------------------------------------------------------
def test_valid_query_parses() -> None:
    q = RetrievalQuery(text="rice scoring", corpus_version="pm-corpus-v1", framework="RICE")
    assert q.top_k == DEFAULT_TOP_K
    assert q.framework is FrameworkName.RICE
    assert q.min_score is None


def test_query_requires_text_and_corpus_version() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(corpus_version="v1")
    with pytest.raises(ValidationError):
        RetrievalQuery(text="x")


def test_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(text="x", corpus_version="v1", unexpected="nope")


def test_citation_requires_content_and_source_title() -> None:
    with pytest.raises(ValidationError):
        _citation(content="   ")  # stripped to empty -> min_length
    with pytest.raises(ValidationError):
        _citation(source_title="")


# --- schema version --------------------------------------------------------
def test_result_default_schema_version() -> None:
    assert _result().schema_version == "1.0"


def test_result_rejects_other_schema_version() -> None:
    with pytest.raises(ValidationError):
        _result(schema_version="2.0")


# --- top_k bounds ----------------------------------------------------------
def test_top_k_bounds() -> None:
    assert RetrievalQuery(text="x", corpus_version="v1", top_k=MAX_TOP_K).top_k == MAX_TOP_K
    with pytest.raises(ValidationError):
        RetrievalQuery(text="x", corpus_version="v1", top_k=0)
    with pytest.raises(ValidationError):
        RetrievalQuery(text="x", corpus_version="v1", top_k=MAX_TOP_K + 1)


# --- min_score bounds ------------------------------------------------------
@pytest.mark.parametrize("value", [-1.0, 0.0, 0.5, 1.0])
def test_min_score_accepts_unit_range(value: float) -> None:
    assert RetrievalQuery(text="x", corpus_version="v1", min_score=value).min_score == value


@pytest.mark.parametrize("value", [-1.0001, 1.0001, 2.0, -5.0])
def test_min_score_rejects_out_of_range(value: float) -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(text="x", corpus_version="v1", min_score=value)


# --- immutability ----------------------------------------------------------
def test_contracts_are_frozen() -> None:
    q = RetrievalQuery(text="x", corpus_version="v1")
    with pytest.raises(ValidationError):
        q.top_k = 9
    r = _result()
    with pytest.raises(ValidationError):
        r.corpus_version = "other"


# --- meta count-partition invariant ----------------------------------------
def test_empty_result_is_valid() -> None:
    result = _result(
        citations=[],
        meta=_meta(vector_match_count=0, resolved_count=0),
    )
    assert result.citations == []


def test_consistent_counts_with_drops() -> None:
    # 3 matched, 1 stale (no row), 2 resolved, 1 score-dropped -> 1 citation.
    result = _result(
        citations=[_citation()],
        meta=_meta(vector_match_count=3, resolved_count=2, dropped_stale_count=1, dropped_score_count=1),
    )
    assert len(result.citations) == 1


def test_resolved_count_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _result(
            citations=[_citation()],
            meta=_meta(vector_match_count=3, resolved_count=3, dropped_stale_count=1),
        )


def test_citation_length_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        _result(
            citations=[_citation(), _citation()],  # 2 citations
            meta=_meta(vector_match_count=2, resolved_count=2, dropped_score_count=1),  # implies 1
        )


# --- serialization ---------------------------------------------------------
def test_result_json_round_trip() -> None:
    result = _result(
        citations=[_citation(framework="JTBD", section="Ch. 2", source_uri="https://x/y")],
        meta=_meta(),
    )
    dumped = result.model_dump(mode="json")
    assert isinstance(dumped["citations"][0]["chunk_id"], str)  # UUID -> str
    assert dumped["citations"][0]["framework"] == "JTBD"  # enum -> value
    assert dumped["citations"][0]["source_type"] == "framework"

    assert RetrievalResult.model_validate(dumped) == result
    assert RetrievalResult.model_validate_json(result.model_dump_json()) == result
