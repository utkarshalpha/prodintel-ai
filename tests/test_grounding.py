"""Unit tests for the Stage 1 grounding gate.

This is the system's primary anti-hallucination boundary, so the tests are
deliberately adversarial: claims that point at unrelated text, out-of-bounds spans,
whitespace-only spans, off-by-one boundaries, and mixed batches where some claims
ground and others do not.
"""

from __future__ import annotations

import pytest

from app.ai_contracts.base import ModelMeta
from app.ai_contracts.validation.grounding import (
    DEFAULT_CLAIM_THRESHOLD,
    GroundingStatus,
    ground_claim,
    ground_claims,
    normalize,
    span_grounding_ratio,
    token_overlap,
    validate_parsed_signal_grounding,
)

from tests.conftest import make_claim, make_parsed_signal


RAW = "The mobile checkout fails when the user taps pay during peak hours."


# --------------------------------------------------------------------------- #
# normalize / token_overlap
# --------------------------------------------------------------------------- #
def test_normalize_lowercases_and_strips_punctuation() -> None:
    assert normalize("High-Priority, NOW!") == ["high", "priority", "now"]


def test_normalize_removes_stopwords_by_default() -> None:
    # "the", "is", "a" are stopwords and should be dropped.
    assert normalize("the checkout is a failure") == ["checkout", "failure"]


def test_normalize_can_keep_stopwords() -> None:
    assert normalize("the checkout", remove_stopwords=False) == ["the", "checkout"]


def test_token_overlap_identical_strings_is_one() -> None:
    assert token_overlap("checkout fails", "checkout fails") == pytest.approx(1.0)


def test_token_overlap_disjoint_strings_is_zero() -> None:
    assert token_overlap("checkout fails", "billing succeeds") == pytest.approx(0.0)


def test_token_overlap_empty_side_is_zero() -> None:
    # Only stopwords -> empty content token set -> 0.0, never a divide error.
    assert token_overlap("the and of", "checkout") == pytest.approx(0.0)


def test_token_overlap_is_symmetric() -> None:
    a, b = "mobile checkout fails", "checkout fails badly"
    assert token_overlap(a, b) == pytest.approx(token_overlap(b, a))


# --------------------------------------------------------------------------- #
# ground_claim -- success
# --------------------------------------------------------------------------- #
def test_grounded_claim_passes() -> None:
    # Span covers "mobile checkout fails"; claim paraphrases it minimally.
    span = (4, 25)
    assert RAW[span[0]:span[1]] == "mobile checkout fails"
    claim = make_claim(text="mobile checkout fails", span=span)
    result = ground_claim(claim, RAW)
    assert result.status is GroundingStatus.GROUNDED
    assert result.grounded is True
    assert result.overlap >= DEFAULT_CLAIM_THRESHOLD
    assert result.matched_text == "mobile checkout fails"


# --------------------------------------------------------------------------- #
# ground_claim -- failure modes
# --------------------------------------------------------------------------- #
def test_out_of_bounds_span_fails_bounds() -> None:
    claim = make_claim(text="anything", span=(0, len(RAW) + 10))
    result = ground_claim(claim, RAW)
    assert result.status is GroundingStatus.FAIL_BOUNDS
    assert result.matched_text == ""
    assert result.overlap == 0.0


def test_whitespace_only_span_fails_empty() -> None:
    text = "abc      def"  # spaces between offsets 3 and 9
    assert text[3:9].strip() == ""
    claim = make_claim(text="def", span=(3, 9))
    result = ground_claim(claim, text)
    assert result.status is GroundingStatus.FAIL_EMPTY


def test_unrelated_span_fails_ungrounded() -> None:
    # Span points at "peak hours" but the claim talks about refunds -> hallucination.
    span = (RAW.index("peak hours"), RAW.index("peak hours") + len("peak hours"))
    claim = make_claim(text="users cannot request a refund", span=span)
    result = ground_claim(claim, RAW)
    assert result.status is GroundingStatus.FAIL_UNGROUNDED
    assert result.overlap < DEFAULT_CLAIM_THRESHOLD


def test_threshold_is_configurable() -> None:
    # A partial-overlap claim that fails at 0.6 can pass at a lenient threshold.
    span = (4, 25)  # "mobile checkout fails"
    claim = make_claim(text="mobile checkout sometimes fails intermittently", span=span)
    strict = ground_claim(claim, RAW, threshold=0.9)
    lenient = ground_claim(claim, RAW, threshold=0.3)
    assert strict.status is GroundingStatus.FAIL_UNGROUNDED
    assert lenient.status is GroundingStatus.GROUNDED


def test_off_by_one_full_text_span_is_in_bounds() -> None:
    # end == len(raw_text) is valid (half-open interval).
    claim = make_claim(text=RAW, span=(0, len(RAW)))
    result = ground_claim(claim, RAW)
    assert result.status is GroundingStatus.GROUNDED


# --------------------------------------------------------------------------- #
# ground_claims / span_grounding_ratio
# --------------------------------------------------------------------------- #
def test_ground_claims_preserves_order_and_pairs() -> None:
    good = make_claim(text="mobile checkout fails", span=(4, 25))
    bad = make_claim(text="refunds are delayed", span=(0, 3))
    pairs = ground_claims([good, bad], RAW)
    assert [c for c, _ in pairs] == [good, bad]
    assert pairs[0][1].grounded is True
    assert pairs[1][1].grounded is False


def test_span_grounding_ratio_mixed_batch() -> None:
    good = make_claim(text="mobile checkout fails", span=(4, 25))
    bad = make_claim(text="refunds are delayed", span=(0, 3))
    assert span_grounding_ratio([good, bad], RAW) == pytest.approx(0.5)


def test_span_grounding_ratio_empty_is_zero() -> None:
    assert span_grounding_ratio([], RAW) == 0.0


# --------------------------------------------------------------------------- #
# validate_parsed_signal_grounding -- the stage gate
# --------------------------------------------------------------------------- #
def test_report_passes_when_any_claim_grounds(model_meta: ModelMeta) -> None:
    good = make_claim(text="mobile checkout fails", span=(4, 25))
    bad = make_claim(text="refunds are delayed", span=(0, 3))
    parsed = make_parsed_signal(model_meta=model_meta, claims=[good, bad])
    report = validate_parsed_signal_grounding(parsed, RAW)
    assert report.passed is True
    assert report.ratio == pytest.approx(0.5)
    assert good in report.grounded_claims
    assert len(report.rejected) == 1
    assert report.rejected[0][0] is bad


def test_report_fails_when_all_claims_ungrounded(model_meta: ModelMeta) -> None:
    bad1 = make_claim(text="refunds are delayed", span=(0, 3))
    bad2 = make_claim(text="billing is duplicated", span=(0, 3))
    parsed = make_parsed_signal(model_meta=model_meta, claims=[bad1, bad2])
    report = validate_parsed_signal_grounding(parsed, RAW)
    assert report.passed is False
    assert report.grounded_claims == []
    assert report.ratio == pytest.approx(0.0)
    assert len(report.rejected) == 2


def test_report_rejected_carries_reason(model_meta: ModelMeta) -> None:
    bad = make_claim(text="refunds are delayed", span=(0, 3))
    parsed = make_parsed_signal(model_meta=model_meta, claims=[bad])
    report = validate_parsed_signal_grounding(parsed, RAW)
    _, result = report.rejected[0]
    assert result.reason  # non-empty, suitable for a retry message or PM surfacing
    assert result.status is GroundingStatus.FAIL_UNGROUNDED
