"""Canonical enumerations for ProdIntel AI.

This module is the **single source of truth** for every closed value set used
across three layers of the system:

1. The PostgreSQL ``ENUM`` types (mirrored 1:1 by name and members).
2. The Pydantic AI-contract models (the LLM trust boundary).
3. The FastAPI request/response schemas (the HTTP boundary).

Defining each enum exactly once here prevents the classic "same enum declared in
three places, drifts in two" bug. Every other module imports from this file; no
module re-declares a value set.

Notes
-----
* ``source_type`` and ``stakeholder_type`` are two distinct PostgreSQL enum types
  that happen to share an identical member set. In Python we model them with a
  single :class:`StakeholderType` to avoid duplication; both the ``signal.source_type``
  column and any stakeholder-typed field map onto it.
* Only :class:`StakeholderType` and :class:`ConfidenceBasis` are exercised by
  Sprint 1 (Stage 1). The remaining enums are included because this file is the
  shared spine for later stages and the database schema, and they must not be
  re-declared elsewhere.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "StakeholderType",
    "ConfidenceBasis",
    "WorkspaceStatus",
    "FeatureStatus",
    "ConflictType",
    "ConflictStatus",
    "SubjectType",
    "Stance",
    "FrameworkName",
    "DecisionStatus",
    "EvidenceType",
    "Relationship",
    "HistoryEvent",
]


class StakeholderType(str, Enum):
    """Origin of a stakeholder signal / the party holding a position.

    Backs both the ``source_type`` and ``stakeholder_type`` PostgreSQL enums,
    which share this member set. Inherits from ``str`` so values serialize as
    plain strings in JSON and compare equal to their string form.
    """

    CUSTOMER = "customer"
    SALES = "sales"
    ENGINEERING = "engineering"
    SUPPORT = "support"
    LEADERSHIP = "leadership"


class ConfidenceBasis(str, Enum):
    """Qualitative band derived deterministically from a numeric confidence score.

    The band is never supplied by the LLM directly; it is always recomputed from
    the score (see :func:`app.ai_contracts.base.confidence_basis_for_score`) so
    the label and the number can never disagree.
    """

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    INSUFFICIENT = "insufficient"


class WorkspaceStatus(str, Enum):
    """Lifecycle state of a decision workspace."""

    DRAFT = "draft"
    ANALYZING = "analyzing"
    SCORED = "scored"
    DECIDED = "decided"


class FeatureStatus(str, Enum):
    """Lifecycle state of an extracted feature."""

    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    MERGED = "merged"


class ConflictType(str, Enum):
    """Dimension along which two stakeholders disagree."""

    PRIORITY = "priority"
    RISK = "risk"
    RESOURCE = "resource"
    STRATEGIC = "strategic"


class ConflictStatus(str, Enum):
    """Resolution state of a detected conflict."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class SubjectType(str, Enum):
    """What a conflict is about."""

    FEATURE = "feature"
    OBJECTIVE = "objective"


class Stance(str, Enum):
    """A stakeholder's position toward a feature or objective."""

    ADVOCATE = "advocate"
    OPPOSE = "oppose"
    RISK_FLAG = "risk_flag"
    NEUTRAL = "neutral"


class FrameworkName(str, Enum):
    """Supported product-management knowledge sources for the RAG corpus."""

    RICE = "RICE"
    JTBD = "JTBD"
    MOSCOW = "MoSCoW"
    STRATEGY = "STRATEGY"
    PRD_TEMPLATE = "PRD_TEMPLATE"


class DecisionStatus(str, Enum):
    """Lifecycle state of a synthesized decision."""

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    OVERRIDDEN = "overridden"
    REJECTED = "rejected"


class EvidenceType(str, Enum):
    """Kind of node a provenance edge points at (polymorphic target)."""

    SIGNAL = "signal"
    PARSED_CLAIM = "parsed_claim"
    SCORE = "score"
    CONFLICT = "conflict"
    FRAMEWORK_CITATION = "framework_citation"


class Relationship(str, Enum):
    """Directional meaning of a provenance edge."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INFORMS = "informs"
    GROUNDS = "grounds"


class HistoryEvent(str, Enum):
    """Auditable event recorded in the append-only decision history."""

    CREATED = "created"
    EDITED = "edited"
    ACCEPTED = "accepted"
    OVERRIDDEN = "overridden"
    REOPENED = "reopened"
