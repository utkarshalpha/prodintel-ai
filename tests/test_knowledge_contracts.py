"""Phase 6A -- structural validation of the knowledge-ingestion contracts.

Confirms the contracts enforce only structural rules (non-empty text, bounded
fields, >= 1 chunk) and reject unknown keys, while leaving computed fields
(content hashes, chroma ids) and semantic rules to the ingestion service.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ai_contracts.enums import FrameworkName, KnowledgeSourceType
from app.ai_contracts.knowledge import KnowledgeChunkContract, KnowledgeSourceContract


def _chunk(**overrides) -> dict:
    base = {"ordinal": 0, "content": "Reach x Impact x Confidence / Effort.", "token_count": 8}
    base.update(overrides)
    return base


def _source(**overrides) -> dict:
    base = {
        "source_type": "framework",
        "framework": "Kano",
        "title": "Kano Model Reference",
        "corpus_version": "pm-corpus-v1",
        "embedding_model_id": "hash-fake-v1",
        "chunks": [_chunk()],
    }
    base.update(overrides)
    return base


def test_valid_source_contract_parses() -> None:
    contract = KnowledgeSourceContract(**_source())
    assert contract.source_type is KnowledgeSourceType.FRAMEWORK
    assert contract.framework is FrameworkName.KANO
    assert contract.chunks[0].section is None
    assert contract.author is None


def test_source_requires_at_least_one_chunk() -> None:
    with pytest.raises(ValidationError):
        KnowledgeSourceContract(**_source(chunks=[]))


def test_chunk_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        KnowledgeChunkContract(**_chunk(content="   "))  # stripped to empty -> min_length fails


def test_chunk_rejects_nonpositive_token_count() -> None:
    with pytest.raises(ValidationError):
        KnowledgeChunkContract(**_chunk(token_count=0))


def test_chunk_rejects_negative_ordinal() -> None:
    with pytest.raises(ValidationError):
        KnowledgeChunkContract(**_chunk(ordinal=-1))


def test_extra_field_is_forbidden() -> None:
    with pytest.raises(ValidationError):
        KnowledgeSourceContract(**_source(content_hash="deadbeef"))  # computed, not accepted


def test_contract_is_frozen() -> None:
    contract = KnowledgeSourceContract(**_source())
    with pytest.raises(ValidationError):
        contract.title = "mutated"


def test_framework_is_optional() -> None:
    contract = KnowledgeSourceContract(**_source(source_type="company_strategy", framework=None))
    assert contract.framework is None
    assert contract.source_type is KnowledgeSourceType.COMPANY_STRATEGY
