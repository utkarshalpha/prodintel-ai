"""Stage 1 grounding gate -- the system's primary anti-hallucination boundary.

The model is *untrusted*. After it returns a :class:`ParsedSignalContract`, every
extracted claim must be proven against the original signal text before it is
allowed to become system state. This module is that proof.

A claim is **grounded** when:

1. its ``source_span`` lies inside the signal's ``raw_text`` (bounds),
2. the slice ``raw_text[start:end]`` is not just whitespace (non-empty), and
3. the slice shares enough tokens with the claim text to substantiate it
   (overlap >= threshold).

All functions here are pure, deterministic, and dependency-free -- no model calls,
no I/O -- which is exactly what a trust boundary should be: cheap, total, and
unit-testable in isolation. The overlap metric is token-set Jaccard similarity,
matching the finalized contract spec. Because the model is instructed to make spans
*exact*, a correctly-anchored claim's text and its span will share almost all tokens
and clear the threshold comfortably; a fabricated claim pointing at unrelated text
will not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.ai_contracts.stage1_signal import ExtractedClaim, ParsedSignalContract

__all__ = [
    "GroundingStatus",
    "GroundingResult",
    "SignalGroundingReport",
    "DEFAULT_CLAIM_THRESHOLD",
    "normalize",
    "token_overlap",
    "ground_claim",
    "ground_claims",
    "span_grounding_ratio",
    "validate_parsed_signal_grounding",
]


# Default minimum token-overlap for a claim to count as grounded. Chosen per the
# finalized AI-contract spec; exposed as a parameter on every public function so
# callers (and the eval harness) can sweep it.
DEFAULT_CLAIM_THRESHOLD = 0.6

# Token pattern: maximal runs of ASCII letters/digits. Punctuation and whitespace
# are separators, so "high-priority!" -> {"high", "priority"}.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimal closed-class stopword set. Removed before overlap so that two strings
# sharing only filler words ("the", "to", "a") do not appear spuriously grounded.
# Deliberately small and explicit -- a large list would risk dropping
# domain-meaningful tokens and is unnecessary for the grounding decision.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "for",
        "with", "as", "at", "by", "is", "are", "was", "were", "be", "been", "being",
        "it", "its", "this", "that", "these", "those", "we", "our", "us", "they",
        "their", "i", "me", "my", "you", "your", "he", "she", "his", "her",
        "do", "does", "did", "have", "has", "had", "will", "would", "can", "could",
        "should", "from", "so", "than", "then", "there", "here",
    }
)


def normalize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Tokenize ``text`` into a normalized list of content tokens.

    Lower-cases, splits on non-alphanumeric boundaries, and (by default) removes a
    small set of stopwords. Returns a list (order preserved) rather than a set so
    callers can inspect token counts; :func:`token_overlap` set-ifies internally.

    Parameters
    ----------
    text:
        Arbitrary input string.
    remove_stopwords:
        When ``True`` (default), closed-class filler words are dropped so they
        cannot inflate overlap.

    Returns
    -------
    list[str]
        Normalized content tokens, possibly empty.
    """

    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [token for token in tokens if token not in _STOPWORDS]
    return tokens


def token_overlap(a: str, b: str, *, remove_stopwords: bool = True) -> float:
    """Token-set Jaccard similarity between two strings, in ``[0, 1]``.

    Defined as ``|tokens(a) & tokens(b)| / |tokens(a) | tokens(b)|``. Returns
    ``0.0`` whenever either side has no content tokens, since an empty side can
    substantiate nothing.

    Parameters
    ----------
    a, b:
        Strings to compare.
    remove_stopwords:
        Forwarded to :func:`normalize`; keep ``True`` for grounding decisions.

    Returns
    -------
    float
        Jaccard similarity in ``[0, 1]``.
    """

    set_a = set(normalize(a, remove_stopwords=remove_stopwords))
    set_b = set(normalize(b, remove_stopwords=remove_stopwords))
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


class GroundingStatus(str, Enum):
    """Outcome of grounding a single claim against its signal text."""

    GROUNDED = "grounded"
    FAIL_BOUNDS = "fail_bounds"
    FAIL_EMPTY = "fail_empty"
    FAIL_UNGROUNDED = "fail_ungrounded"


@dataclass(frozen=True)
class GroundingResult:
    """Result of grounding one :class:`ExtractedClaim`.

    Attributes
    ----------
    status:
        The :class:`GroundingStatus` outcome.
    overlap:
        Token-overlap score actually computed (``0.0`` when grounding failed before
        the overlap step).
    matched_text:
        The ``raw_text`` slice referenced by the claim's span (empty when the span
        was out of bounds).
    reason:
        Human-readable explanation, suitable for surfacing to a PM or feeding a
        deterministic retry message back to the model.
    """

    status: GroundingStatus
    overlap: float
    matched_text: str
    reason: str

    @property
    def grounded(self) -> bool:
        """``True`` iff this claim passed the grounding gate."""

        return self.status is GroundingStatus.GROUNDED


def ground_claim(
    claim: ExtractedClaim,
    raw_text: str,
    *,
    threshold: float = DEFAULT_CLAIM_THRESHOLD,
) -> GroundingResult:
    """Verify that ``claim`` is substantiated by its span in ``raw_text``.

    Implements the finalized grounding algorithm, short-circuiting on the first
    failure:

    1. **Bounds** -- ``0 <= start < end <= len(raw_text)``; else
       :attr:`GroundingStatus.FAIL_BOUNDS`.
    2. **Non-empty** -- ``raw_text[start:end]`` must contain non-whitespace; else
       :attr:`GroundingStatus.FAIL_EMPTY`.
    3. **Overlap** -- token overlap between the claim text and the slice must be
       ``>= threshold``; else :attr:`GroundingStatus.FAIL_UNGROUNDED`.

    Never raises on claim/text content -- it returns a :class:`GroundingResult` so
    the caller decides whether to drop the claim, retry, or fail the stage.

    Parameters
    ----------
    claim:
        The claim to verify. Its ``source_span`` is already internally well-ordered
        (guaranteed by :class:`ExtractedClaim`); this function checks it against the
        actual text length.
    raw_text:
        The original, immutable signal text the claim was extracted from.
    threshold:
        Minimum token overlap to count as grounded (default
        :data:`DEFAULT_CLAIM_THRESHOLD`).

    Returns
    -------
    GroundingResult
        Outcome plus the computed overlap, matched slice, and a reason string.
    """

    start, end = claim.source_span
    text_length = len(raw_text)

    # Step 1: bounds. start >= 0 and end > start are guaranteed by the contract,
    # but end may exceed this particular text, so we re-check defensively.
    if start < 0 or end > text_length or start >= end:
        return GroundingResult(
            status=GroundingStatus.FAIL_BOUNDS,
            overlap=0.0,
            matched_text="",
            reason=(
                f"span [{start}, {end}) is out of bounds for raw_text of "
                f"length {text_length}"
            ),
        )

    matched_text = raw_text[start:end]

    # Step 2: the referenced slice must carry content.
    if not matched_text.strip():
        return GroundingResult(
            status=GroundingStatus.FAIL_EMPTY,
            overlap=0.0,
            matched_text=matched_text,
            reason=f"span [{start}, {end}) references only whitespace",
        )

    # Step 3: the slice must lexically substantiate the claim.
    overlap = token_overlap(claim.text, matched_text)
    if overlap < threshold:
        return GroundingResult(
            status=GroundingStatus.FAIL_UNGROUNDED,
            overlap=overlap,
            matched_text=matched_text,
            reason=(
                f"token overlap {overlap:.3f} between claim and span is below "
                f"threshold {threshold:.3f}"
            ),
        )

    return GroundingResult(
        status=GroundingStatus.GROUNDED,
        overlap=overlap,
        matched_text=matched_text,
        reason="claim substantiated by its source span",
    )


def ground_claims(
    claims: list[ExtractedClaim],
    raw_text: str,
    *,
    threshold: float = DEFAULT_CLAIM_THRESHOLD,
) -> list[tuple[ExtractedClaim, GroundingResult]]:
    """Ground every claim in ``claims`` against ``raw_text``.

    Returns a list of ``(claim, result)`` pairs in input order, so callers can both
    keep the grounded claims and report precisely why others were rejected.
    """

    return [(claim, ground_claim(claim, raw_text, threshold=threshold)) for claim in claims]


def span_grounding_ratio(
    claims: list[ExtractedClaim],
    raw_text: str,
    *,
    threshold: float = DEFAULT_CLAIM_THRESHOLD,
) -> float:
    """Fraction of ``claims`` that pass the grounding gate, in ``[0, 1]``.

    This deterministic ratio is the ``span_grounding_ratio`` confidence component
    fed downstream (into Reasoning Quality). Returns ``0.0`` for an empty claim
    list -- nothing grounded means no grounding.
    """

    if not claims:
        return 0.0
    grounded = sum(1 for _, result in ground_claims(claims, raw_text, threshold=threshold) if result.grounded)
    return grounded / len(claims)


@dataclass(frozen=True)
class SignalGroundingReport:
    """Aggregate grounding outcome for one :class:`ParsedSignalContract`.

    Attributes
    ----------
    ratio:
        ``span_grounding_ratio`` over all claims (grounded / emitted).
    grounded_claims:
        The claims that passed the gate (safe to persist).
    rejected:
        ``(claim, result)`` pairs that failed, with the reason for each.
    passed:
        ``True`` iff at least one claim survived. A parse where every claim is
        dropped fails Stage 1 entirely and must not produce a feature.
    """

    ratio: float
    grounded_claims: list[ExtractedClaim]
    rejected: list[tuple[ExtractedClaim, GroundingResult]]

    @property
    def passed(self) -> bool:
        """Stage passes only if at least one claim is grounded."""

        return len(self.grounded_claims) >= 1


def validate_parsed_signal_grounding(
    parsed: ParsedSignalContract,
    raw_text: str,
    *,
    threshold: float = DEFAULT_CLAIM_THRESHOLD,
) -> SignalGroundingReport:
    """Run the full Stage 1 grounding gate over a parsed signal.

    Partitions the contract's claims into grounded vs. rejected, computes the
    grounding ratio, and reports whether the stage passes (>= 1 grounded claim).
    This is the function the Signal service calls right after schema validation and
    before persisting anything.

    Parameters
    ----------
    parsed:
        The schema-valid Stage 1 contract returned by the model.
    raw_text:
        The original immutable text of the signal identified by
        ``parsed.signal_id``. The caller is responsible for fetching the matching
        text; this function does not perform I/O.
    threshold:
        Grounding overlap threshold (default :data:`DEFAULT_CLAIM_THRESHOLD`).

    Returns
    -------
    SignalGroundingReport
        Grounded claims, rejected claims with reasons, the ratio, and pass/fail.
    """

    grounded_claims: list[ExtractedClaim] = []
    rejected: list[tuple[ExtractedClaim, GroundingResult]] = []

    for claim, result in ground_claims(parsed.extracted_claims, raw_text, threshold=threshold):
        if result.grounded:
            grounded_claims.append(claim)
        else:
            rejected.append((claim, result))

    total = len(parsed.extracted_claims)
    ratio = (len(grounded_claims) / total) if total else 0.0

    return SignalGroundingReport(
        ratio=ratio,
        grounded_claims=grounded_claims,
        rejected=rejected,
    )
