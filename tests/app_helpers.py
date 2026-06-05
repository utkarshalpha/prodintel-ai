"""Test doubles and fixtures for the application layer (DB + service + API).

Provides an in-memory SQLite engine (shared across sessions via StaticPool), a
session factory, and ``Stage1FakeClient`` -- a deterministic, network-free
``ToolCallClient`` that echoes the signal_id from the prompt and emits either a
grounded or an ungrounded claim, so the full analyze flow can be exercised.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401  -- registers tables on Base.metadata
from app.ai_runtime.interfaces import LLMMessage, LLMToolResponse, ToolSpec
from app.db.base import Base
from app.stages.stage1 import prompts
from app.stages.stage2 import prompts as stage2_prompts
from app.stages.stage3 import prompts as stage3_prompts

_SIGNAL_ID_RE = re.compile(r"signal_id:\s*([0-9a-fA-F-]{36})")
_RAW_RE = re.compile(r"<<<SIGNAL>>>\n(.*)\n<<<END>>>", re.DOTALL)
_SOURCE_HINT_RE = re.compile(r"Known source channel \(hint\):\s*(\w+)")
_FEATURE_ID_RE = re.compile(r"feature_id:\s*([0-9a-fA-F-]{36})")
_SIGNAL_STAKEHOLDER_RE = re.compile(r"signal_id:\s*([0-9a-fA-F-]{36})\s*\|\s*stakeholder:\s*(\w+)")


def make_engine():
    """Create a fresh in-memory SQLite engine with all tables created."""

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )

    # SQLite does not enforce foreign keys unless asked; enable so ON DELETE
    # RESTRICT/CASCADE behave like production Postgres.
    @event.listens_for(engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, _record):  # pragma: no cover - trivial
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine


def make_session_factory(engine) -> sessionmaker:
    """Session factory bound to ``engine`` (expire_on_commit off for post-commit reads)."""

    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)


class Stage1FakeClient:
    """Deterministic Stage 1 client: echoes the signal_id and grounds (or not).

    With ``grounded=True`` it returns a claim whose text is the entire signal and
    whose span covers the whole text -- guaranteed to pass the grounding gate. With
    ``grounded=False`` it returns an unrelated claim anchored to the first few
    characters, which always fails grounding (used to exercise the failure path).
    """

    def __init__(
        self,
        *,
        grounded: bool = True,
        model_id: str = "claude-test",
        input_tokens: int = 120,
        output_tokens: int = 30,
    ) -> None:
        self._grounded = grounded
        self._model_id = model_id
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.calls = 0

    def complete(self, *, system: str, messages: Sequence[LLMMessage], tool: ToolSpec) -> LLMToolResponse:
        self.calls += 1
        user_prompt = messages[0].content  # the original signal prompt
        signal_id = _SIGNAL_ID_RE.search(user_prompt).group(1)  # type: ignore[union-attr]
        raw = _RAW_RE.search(user_prompt).group(1)  # type: ignore[union-attr]

        # Classify stakeholder_type from the source-channel hint when present, so the
        # downstream pipeline (Stage 2/3) sees realistic, distinct stakeholders.
        hint = _SOURCE_HINT_RE.search(user_prompt)
        stakeholder_type = hint.group(1) if hint else "customer"

        if self._grounded:
            claim = {"text": raw, "source_span": [0, len(raw)], "claim_confidence": 0.95}
        else:
            claim = {
                "text": "zzz qqq www totally unrelated tokens",
                "source_span": [0, min(3, len(raw))],
                "claim_confidence": 0.9,
            }

        payload: dict[str, Any] = {
            "signal_id": signal_id,
            "intent": "Stakeholder intent extracted from the signal",
            "stakeholder_type": stakeholder_type,
            "urgency": 3,
            "sentiment": 0.0,
            "extracted_claims": [claim],
            "confidence": {"score": 0.8, "components": {"span_grounding_ratio": 1.0}},
        }
        return LLMToolResponse(
            tool_name=prompts.STAGE1_TOOL_NAME,
            tool_input=payload,
            stop_reason="tool_use",
            model_id=self._model_id,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )


class Stage2FakeClient:
    """Deterministic Stage 2 client: clusters the prompt's signals into features.

    Modes:
    * ``cluster_all``  -- one feature whose source_signal_ids are every input id
                          (provenance passes).
    * ``fabricate``    -- one feature citing a signal id not in the input
                          (provenance fails: unknown_signal).
    * ``drop_one``     -- one feature citing only the first input id, others neither
                          assigned nor unassigned (provenance fails: missing).
    """

    def __init__(
        self,
        *,
        mode: str = "cluster_all",
        model_id: str = "claude-test",
        input_tokens: int = 140,
        output_tokens: int = 60,
    ) -> None:
        self._mode = mode
        self._model_id = model_id
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.calls = 0

    def complete(self, *, system: str, messages: Sequence[LLMMessage], tool: ToolSpec) -> LLMToolResponse:
        self.calls += 1
        signal_ids = _SIGNAL_ID_RE.findall(messages[0].content)

        if self._mode == "fabricate":
            sources = ["00000000-0000-4000-8000-000000000000"]
            unassigned = list(signal_ids)
        elif self._mode == "drop_one":
            sources = signal_ids[:1]
            unassigned = []
        else:  # cluster_all
            sources = list(signal_ids)
            unassigned = []

        payload: dict[str, Any] = {
            "features": [
                {
                    "feature_id": "f1",
                    "title": "Reliable mobile checkout",
                    "description": "Ensure mobile checkout completes payment without failure.",
                    "jtbd": "When I check out on mobile, I want payment to succeed, so I can complete my purchase",
                    "source_signal_ids": sources,
                    "confidence": {"score": 0.82, "components": {"cluster_cohesion": 0.82}},
                }
            ],
            "unassigned_signal_ids": unassigned,
        }
        return LLMToolResponse(
            tool_name=stage2_prompts.STAGE2_TOOL_NAME,
            tool_input=payload,
            stop_reason="tool_use",
            model_id=self._model_id,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )


class Stage3FakeClient:
    """Deterministic Stage 3 client: surfaces a conflict between the first two signals.

    Modes:
    * ``conflict`` -- one conflict on the first feature, with the first two signals'
                      stakeholders taking opposing stances (advocate vs risk_flag).
    * ``none``     -- empty conflicts list (stakeholders agree).
    * ``fabricate``-- one conflict citing an evidence signal not in the input
                      (integrity fails: unknown_evidence).
    * ``no_opposition`` -- both positions advocate (integrity fails: missing_opposition).
    """

    def __init__(
        self,
        *,
        mode: str = "conflict",
        model_id: str = "claude-test",
        input_tokens: int = 160,
        output_tokens: int = 70,
    ) -> None:
        self._mode = mode
        self._model_id = model_id
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.calls = 0

    def complete(self, *, system: str, messages: Sequence[LLMMessage], tool: ToolSpec) -> LLMToolResponse:
        self.calls += 1
        content = messages[0].content
        feature_ids = _FEATURE_ID_RE.findall(content)
        pairs = _SIGNAL_STAKEHOLDER_RE.findall(content)  # [(signal_id, stakeholder), ...]

        if self._mode == "none":
            payload: dict[str, Any] = {"conflicts": []}
            return self._respond(payload)

        (s0, st0), (s1, st1) = pairs[0], pairs[1]
        if self._mode == "fabricate":
            evidence0 = ["00000000-0000-4000-8000-000000000000"]
        else:
            evidence0 = [s0]
        stance1 = "advocate" if self._mode == "no_opposition" else "risk_flag"

        conflict = {
            "conflict_id": "c1",
            "conflict_type": "risk",
            "severity": 4,
            "subject_type": "feature",
            "subject_id": feature_ids[0],
            "stakeholders": [st0, st1],
            "positions": [
                {
                    "stakeholder": st0,
                    "stance": "advocate",
                    "summary": f"{st0} advocates for the feature",
                    "evidence_signal_ids": evidence0,
                },
                {
                    "stakeholder": st1,
                    "stance": stance1,
                    "summary": f"{st1} position on the feature",
                    "evidence_signal_ids": [s1],
                },
            ],
            "evidence_signal_ids": evidence0 + [s1],
            "confidence": {"score": 0.78, "components": {"opposition_strength": 0.8}},
        }
        return self._respond({"conflicts": [conflict]})

    def _respond(self, payload: dict[str, Any]) -> LLMToolResponse:
        return LLMToolResponse(
            tool_name=stage3_prompts.STAGE3_TOOL_NAME,
            tool_input=payload,
            stop_reason="tool_use",
            model_id=self._model_id,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )
