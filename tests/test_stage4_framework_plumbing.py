"""Phase 7C-i -- framework-context plumbing into Stage 4.

Covers the three plumbing seams (and nothing downstream of them -- no RetrievalService,
DecisionService loop, repository, or persistence):

* ``Stage4Context.framework_knowledge`` + the ``input_framework_chunk_ids`` helper;
* the prompt rendering of the framework pool (deterministic, ids visible, subset rule);
* the validator adapter feeding ``input_framework_chunk_ids`` into the integrity gate
  and mapping ``unknown_framework_citations`` to a harness issue.

Backward compatibility is the load-bearing requirement: an empty pool must render a
prompt byte-identical to the framework-free Stage 4 and pass the validator exactly as
before, so the legacy flow is provably unchanged.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.ai_contracts.base import ModelMeta
from app.ai_contracts.stage4_decision import DecisionContract, DecisionSynthesisContract
from app.stages.stage4.prompts import (
    ConflictForDecision,
    FeatureForDecision,
    FrameworkPassage,
    SignalForDecision,
    build_user_prompt,
)
from app.stages.stage4.runner import Stage4Context
from app.stages.stage4.validators import DECISION_ISSUE_CODES, DecisionIntegrityValidator

F1 = uuid4()
S1 = uuid4()
K1, K2, K_UNKNOWN = uuid4(), uuid4(), uuid4()


def _feature() -> FeatureForDecision:
    return FeatureForDecision(feature_id=F1, title="Enterprise SSO", jtbd="When..., I want..., so I can...")


def _signal() -> SignalForDecision:
    return SignalForDecision(signal_id=S1, stakeholder_type="sales", intent="want SSO", claims=["SSO critical"])


def _passages() -> list[FrameworkPassage]:
    return [
        FrameworkPassage(
            chunk_id=K1, content="RICE scoring weighs reach, impact, confidence, effort.",
            framework="RICE", source_title="RICE Reference", retrieval_score=0.91,
        ),
        FrameworkPassage(
            chunk_id=K2, content="A book passage with no framework attribution.",
            framework=None, source_title="Inspired", retrieval_score=0.42,
        ),
    ]


def _ctx(framework_knowledge=()) -> Stage4Context:
    return Stage4Context(
        features=[_feature()],
        signals=[_signal()],
        conflicts=[],
        framework_knowledge=list(framework_knowledge),
    )


def _decision(*, framework=()) -> DecisionContract:
    return DecisionContract(
        decision_id="d1",
        subject_type="feature",
        subject_id=F1,
        recommendation="build_now",
        title="Build it",
        rationale="The evidence supports building it; grounded in RICE.",
        priority_rank=1,
        acknowledged_conflict_ids=[],
        evidence_signal_ids=[S1],
        framework_citation_ids=list(framework),
        confidence={"score": 0.8},
    )


def _synthesis(decision: DecisionContract, model_meta: ModelMeta) -> DecisionSynthesisContract:
    return DecisionSynthesisContract(decisions=[decision], model_meta=model_meta)


# --------------------------------------------------------------------------- #
# 1. FrameworkPassage view model + helper
# --------------------------------------------------------------------------- #
def test_framework_passage_is_immutable() -> None:
    passage = _passages()[0]
    with pytest.raises(Exception):  # frozen model -> ValidationError on assignment
        passage.retrieval_score = 0.0


def test_framework_chunk_ids_extracted_in_order() -> None:
    ctx = _ctx(_passages())
    assert ctx.input_framework_chunk_ids == [K1, K2]


def test_empty_pool_yields_no_framework_chunk_ids() -> None:
    assert _ctx().input_framework_chunk_ids == []


# --------------------------------------------------------------------------- #
# 2. Prompt rendering
# --------------------------------------------------------------------------- #
def test_prompt_contains_framework_passages_and_ids() -> None:
    prompt = build_user_prompt([_feature()], [_signal()], [], _passages())
    assert "Framework knowledge" in prompt
    assert str(K1) in prompt and str(K2) in prompt  # chunk ids visible to the model
    assert "RICE scoring weighs reach" in prompt
    assert "framework_citation_ids MUST be one of the" in prompt  # explicit subset rule


def test_prompt_renders_null_framework_attribution() -> None:
    prompt = build_user_prompt([_feature()], [_signal()], [], _passages())
    assert "(unspecified)" in prompt  # K2 has framework=None


def test_prompt_ordering_is_deterministic_and_order_preserving() -> None:
    passages = _passages()
    first = build_user_prompt([_feature()], [_signal()], [], passages)
    second = build_user_prompt([_feature()], [_signal()], [], passages)
    assert first == second  # same pool -> identical text
    assert first.index(str(K1)) < first.index(str(K2))  # rendered in list order

    # Reversing the input pool reverses the rendered order (rendering is order-preserving,
    # so deterministic ordering is the caller/service's contract, faithfully honored here).
    reversed_prompt = build_user_prompt([_feature()], [_signal()], [], list(reversed(passages)))
    assert reversed_prompt.index(str(K2)) < reversed_prompt.index(str(K1))


# --------------------------------------------------------------------------- #
# 3. Backward compatibility -- empty pool == framework-free Stage 4
# --------------------------------------------------------------------------- #
def test_empty_pool_prompt_is_byte_identical_to_legacy() -> None:
    """The empty-pool prompt equals the prompt built with no framework argument at all."""

    legacy = build_user_prompt([_feature()], [_signal()], [])
    with_empty = build_user_prompt([_feature()], [_signal()], [], [])
    assert with_empty == legacy
    assert "Framework knowledge" not in with_empty  # no framework section is emitted


def test_runner_prompt_unchanged_when_no_framework_knowledge() -> None:
    """A Stage4Context without a pool drives exactly the legacy user prompt."""

    from app.stages.stage4.runner import build_stage4_runner
    from tests.app_helpers import Stage4FakeClient

    runner = build_stage4_runner(Stage4FakeClient())
    ctx = _ctx()  # no framework_knowledge
    assert runner.build_user_prompt(ctx) == build_user_prompt([_feature()], [_signal()], [])


# --------------------------------------------------------------------------- #
# 4. Validator adapter wiring
# --------------------------------------------------------------------------- #
def test_validator_receives_framework_ids_and_passes_when_grounded(model_meta: ModelMeta) -> None:
    ctx = _ctx(_passages())
    result = DecisionIntegrityValidator().validate(_synthesis(_decision(framework=(K1,)), model_meta), ctx)
    assert result.ok is True


def test_validator_maps_unknown_framework_citation(model_meta: ModelMeta) -> None:
    ctx = _ctx(_passages())
    synthesis = _synthesis(_decision(framework=(K1, K_UNKNOWN)), model_meta)
    result = DecisionIntegrityValidator().validate(synthesis, ctx)

    assert result.ok is False
    offending = [i for i in result.errors if i.code == DECISION_ISSUE_CODES["unknown_framework_citation"]]
    assert len(offending) == 1
    issue = offending[0]
    assert str(K_UNKNOWN) in issue.message
    assert issue.field == "decisions.0.framework_citation_ids"
    assert "subset" in issue.hint  # framework-specific retry guidance


def test_validator_empty_pool_rejects_any_framework_citation(model_meta: ModelMeta) -> None:
    """With no pool injected, any framework citation is unknown (cannot ground on nothing)."""

    ctx = _ctx()  # empty pool
    result = DecisionIntegrityValidator().validate(_synthesis(_decision(framework=(K1,)), model_meta), ctx)
    assert result.ok is False
    assert any(i.code == DECISION_ISSUE_CODES["unknown_framework_citation"] for i in result.errors)


def test_validator_legacy_pass_unaffected_by_framework_wiring(model_meta: ModelMeta) -> None:
    """A sound, framework-less decision over an empty-pool context still passes cleanly."""

    ctx = _ctx()
    result = DecisionIntegrityValidator().validate(_synthesis(_decision(), model_meta), ctx)
    assert result.ok is True
    assert result.errors == ()
