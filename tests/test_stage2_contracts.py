"""Contract-validation tests for Stage 2 (Feature Extraction)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ai_contracts.base import ModelMeta
from app.ai_contracts.stage2_feature import FeatureContract, FeatureExtractionContract


def _item(**overrides):
    data = {
        "feature_id": "f1",
        "title": "Reliable mobile checkout",
        "description": "Ensure mobile checkout completes payment.",
        "jtbd": "When I check out on mobile, I want payment to succeed, so I can buy",
        "source_signal_ids": [uuid4()],
        "confidence": {"score": 0.8},
    }
    data.update(overrides)
    return FeatureContract(**data)


def test_valid_feature_item(model_meta: ModelMeta) -> None:
    item = _item()
    assert item.title == "Reliable mobile checkout"
    assert len(item.source_signal_ids) == 1


def test_feature_requires_at_least_one_source_signal() -> None:
    with pytest.raises(ValidationError):
        _item(source_signal_ids=[])


def test_feature_title_max_length() -> None:
    with pytest.raises(ValidationError):
        _item(title="x" * 121)


def test_feature_jtbd_min_length() -> None:
    with pytest.raises(ValidationError):
        _item(jtbd="short")


def test_extraction_requires_at_least_one_feature(model_meta: ModelMeta) -> None:
    with pytest.raises(ValidationError):
        FeatureExtractionContract(features=[], model_meta=model_meta)


def test_extraction_defaults_unassigned_to_empty(model_meta: ModelMeta) -> None:
    contract = FeatureExtractionContract(features=[_item()], model_meta=model_meta)
    assert contract.unassigned_signal_ids == []
    assert contract.schema_version == "1.0"


def test_extraction_rejects_unknown_field(model_meta: ModelMeta) -> None:
    with pytest.raises(ValidationError):
        FeatureExtractionContract(features=[_item()], model_meta=model_meta, surprise="x")
