"""Phase L4 -- ingestion & validation.

Covers the pure validators, every parser (manual/CSV/TXT, and gated PDF/DOCX), the
partial-success contract, determinism, source-level errors, the architecture-isolation
guard (subprocess), and the end-to-end proof that ingestion output drives PipelineService.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from app.ai_contracts.enums import StakeholderType as S
from app.ingestion import (
    MAX_TEXT_LENGTH,
    EmptyBatchError,
    IngestionFormat,
    MissingColumnError,
    MissingDependencyError,
    RejectReason,
    UnsupportedFormatError,
    WarningKind,
    ingest,
    ingest_csv,
    ingest_docx,
    ingest_manual,
    ingest_pdf,
    ingest_text,
)
from app.ingestion.contract import FeedbackEntry
from app.ingestion.errors import CorruptDocumentError, FileTooLargeError
from app.ingestion.validators import coerce_stakeholder, normalize_text

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _kinds(result):
    return {w.kind for w in result.warnings}


def _reasons(result):
    return [r.reason for r in result.rejected]


# --------------------------------------------------------------------------- #
# Pure validators
# --------------------------------------------------------------------------- #
def test_normalize_text_strips() -> None:
    assert normalize_text("  hi there  ") == "hi there"
    assert normalize_text("\n\t ") == ""
    assert normalize_text(None) == ""  # type: ignore[arg-type]


def test_coerce_stakeholder_is_case_insensitive_and_safe() -> None:
    assert coerce_stakeholder("Sales") is S.SALES
    assert coerce_stakeholder("  ENGINEERING ") is S.ENGINEERING
    assert coerce_stakeholder(S.SUPPORT) is S.SUPPORT
    assert coerce_stakeholder("marketing") is None
    assert coerce_stakeholder("") is None
    assert coerce_stakeholder(None) is None


# --------------------------------------------------------------------------- #
# Manual
# --------------------------------------------------------------------------- #
def test_manual_happy() -> None:
    r = ingest_manual([("sales", "We need SSO to close Acme."),
                       (S.ENGINEERING, "SSO is risky and slow to build.")])
    assert r.accepted_count == 2 and r.rejected_count == 0
    assert [e.source_ref for e in r.entries] == ["manual:1", "manual:2"]
    assert r.entries[0].stakeholder_type is S.SALES
    assert r.source_format is IngestionFormat.MANUAL


def test_manual_empty_batch_raises() -> None:
    with pytest.raises(EmptyBatchError):
        ingest_manual([])


def test_manual_dedupe_warns_and_drops() -> None:
    r = ingest_manual([("sales", "Same feedback text."), ("sales", "Same feedback text.")])
    assert r.accepted_count == 1
    assert WarningKind.DUPLICATE in _kinds(r)


def test_manual_unknown_stakeholder_rejected_by_default() -> None:
    r = ingest_manual([("marketing", "Growth wants referral loops.")])
    assert r.accepted_count == 0
    assert _reasons(r) == [RejectReason.UNKNOWN_STAKEHOLDER]


def test_manual_coerce_unknown_to_default_warns() -> None:
    r = ingest_manual([("marketing", "Growth wants referral loops.")],
                      default_stakeholder=S.LEADERSHIP, coerce_unknown=True)
    assert r.accepted_count == 1
    assert r.entries[0].stakeholder_type is S.LEADERSHIP
    assert WarningKind.DEFAULTED_STAKEHOLDER in _kinds(r)


def test_low_content_warns_but_accepts() -> None:
    r = ingest_manual([("sales", "!!!")])
    assert r.accepted_count == 1
    assert WarningKind.LOW_CONTENT in _kinds(r)


def test_length_cap_rejects() -> None:
    r = ingest_manual([("sales", "x" * (MAX_TEXT_LENGTH + 1))])
    assert _reasons(r) == [RejectReason.TEXT_TOO_LONG]


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def test_csv_partial_success_and_row_numbers() -> None:
    data = ("stakeholder,feedback\n"
            "sales,We need SSO now.\n"
            "engineering,SSO is risky.\n"
            "marketing,Growth idea.\n"            # unknown stakeholder -> reject
            "sales, \n")                          # blank text -> reject
    r = ingest_csv(data, default_stakeholder=None)
    assert r.accepted_count == 2 and r.rejected_count == 2
    assert [e.source_ref for e in r.entries] == ["upload.csv:row 2", "upload.csv:row 3"]
    assert _reasons(r) == [RejectReason.UNKNOWN_STAKEHOLDER, RejectReason.EMPTY_TEXT]


def test_csv_missing_text_column_raises() -> None:
    with pytest.raises(MissingColumnError):
        ingest_csv("stakeholder,comment\nsales,hello there\n")


def test_csv_header_only_is_empty_batch() -> None:
    with pytest.raises(EmptyBatchError):
        ingest_csv("stakeholder,feedback\n")


def test_csv_custom_columns_and_default_stakeholder() -> None:
    data = "who,note\n,An anonymous but useful comment.\n"
    r = ingest_csv(data, text_column="note", stakeholder_column="who", default_stakeholder=S.CUSTOMER)
    assert r.accepted_count == 1 and r.entries[0].stakeholder_type is S.CUSTOMER


def test_csv_latin1_fallback_warns() -> None:
    data = "stakeholder,feedback\nsales,caf\xe9 feature request\n".encode("latin-1")
    r = ingest_csv(data, default_stakeholder=None)
    assert r.accepted_count == 1
    assert WarningKind.DECODE_FALLBACK in _kinds(r)


def test_csv_bom_is_tolerated() -> None:
    data = "﻿stakeholder,feedback\nsales,SSO matters\n".encode("utf-8")
    r = ingest_csv(data, default_stakeholder=None)
    assert r.accepted_count == 1


# --------------------------------------------------------------------------- #
# TXT
# --------------------------------------------------------------------------- #
def test_txt_paragraph_mode() -> None:
    r = ingest_text("We need dark mode.\n\nIt would hurt the timeline.", default_stakeholder=S.CUSTOMER)
    assert r.accepted_count == 2
    assert [e.source_ref for e in r.entries] == ["upload.txt:¶1", "upload.txt:¶2"]


def test_txt_line_mode() -> None:
    r = ingest_text("first line of feedback\nsecond line of feedback", default_stakeholder=S.SUPPORT,
                    granularity="line")
    assert r.accepted_count == 2
    assert r.entries[0].source_ref == "upload.txt:line 1"


def test_txt_document_mode() -> None:
    r = ingest_text("para one here\n\npara two here", default_stakeholder=S.SUPPORT, granularity="document")
    assert r.accepted_count == 1 and r.entries[0].source_ref == "upload.txt"


def test_txt_blank_is_empty_batch() -> None:
    with pytest.raises(EmptyBatchError):
        ingest_text("   \n\n  \t ", default_stakeholder=S.CUSTOMER)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def test_dispatch_by_extension() -> None:
    r = ingest("stakeholder,feedback\nsales,hi there team\n", filename="x.csv", default_stakeholder=None)
    assert r.source_format is IngestionFormat.CSV


def test_dispatch_explicit_manual_format() -> None:
    r = ingest([("sales", "a useful comment here")], format="manual")
    assert r.source_format is IngestionFormat.MANUAL and r.accepted_count == 1


def test_dispatch_unsupported_extension_raises() -> None:
    with pytest.raises(UnsupportedFormatError):
        ingest(b"...", filename="notes.rtf", default_stakeholder=S.CUSTOMER)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_determinism_identical_result() -> None:
    data = "stakeholder,feedback\nsales,We need SSO.\nengineering,SSO is risky.\n"
    assert ingest_csv(data, default_stakeholder=None) == ingest_csv(data, default_stakeholder=None)


# --------------------------------------------------------------------------- #
# PDF (gated on pypdf) + missing-dependency path (always runs)
# --------------------------------------------------------------------------- #
def test_pdf_too_large_rejected() -> None:
    with pytest.raises(FileTooLargeError):
        ingest_pdf(b"0" * (5 * 1024 * 1024 + 1), default_stakeholder=S.CUSTOMER)


def test_pdf_missing_dependency(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pypdf", None)  # force ImportError on `import pypdf`
    with pytest.raises(MissingDependencyError):
        ingest_pdf(b"%PDF-1.4", default_stakeholder=S.CUSTOMER)


def test_pdf_blank_page_has_no_text() -> None:
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(CorruptDocumentError):
        ingest_pdf(buf.getvalue(), default_stakeholder=S.CUSTOMER)


def test_pdf_corrupt_bytes() -> None:
    pytest.importorskip("pypdf")
    with pytest.raises(CorruptDocumentError):
        ingest_pdf(b"this is definitely not a pdf file", default_stakeholder=S.CUSTOMER)


# --------------------------------------------------------------------------- #
# DOCX (gated on python-docx) + missing-dependency path (always runs)
# --------------------------------------------------------------------------- #
def test_docx_missing_dependency(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "docx", None)  # force ImportError on `import docx`
    with pytest.raises(MissingDependencyError):
        ingest_docx(b"PK\x03\x04", default_stakeholder=S.CUSTOMER)


def test_docx_roundtrip_happy() -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("We urgently need dark mode.")
    document.add_paragraph("")  # empty -> skipped
    document.add_paragraph("Dark mode is risky to build.")
    buf = io.BytesIO()
    document.save(buf)
    r = ingest_docx(buf.getvalue(), default_stakeholder=S.CUSTOMER)
    assert r.accepted_count == 2
    assert [e.source_ref for e in r.entries] == ["upload.docx:¶1", "upload.docx:¶2"]


def test_docx_corrupt_bytes() -> None:
    pytest.importorskip("docx")
    with pytest.raises(CorruptDocumentError):
        ingest_docx(b"not a real docx zip", default_stakeholder=S.CUSTOMER)


# --------------------------------------------------------------------------- #
# Architecture isolation guard (subprocess: deterministic regardless of suite order)
# --------------------------------------------------------------------------- #
def test_ingestion_imports_are_isolated() -> None:
    code = (
        "import importlib, sys\n"
        "importlib.import_module('app.ingestion')\n"
        "forbidden = ['app.api', 'app.stages', 'streamlit', 'app.services.pipeline_service', 'app.ai_clients']\n"
        "leaked = sorted(m for m in forbidden if any(k == m or k.startswith(m + '.') for k in list(sys.modules)))\n"
        "print('LEAKED=' + ','.join(leaked))\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO_ROOT),
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    leaked = res.stdout.strip().split("LEAKED=")[-1]
    assert leaked == "", f"app.ingestion leaked forbidden imports: {leaked}"


# --------------------------------------------------------------------------- #
# Integration proof: ingest_csv -> FeedbackEntry[] -> PipelineService -> LocalHeuristicClient
# --------------------------------------------------------------------------- #
def test_ingestion_drives_pipeline_end_to_end() -> None:
    import app.api.app  # noqa: F401  -- init api package (explanation service imports it)
    from app.ai_clients.local_heuristic_client import LocalHeuristicClient
    from app.repositories.conflict_repository import ConflictRepository
    from app.repositories.decision_repository import DecisionRepository
    from app.repositories.feature_repository import FeatureRepository
    from app.repositories.signal_repository import SignalRepository
    from app.services.decision_explanation_service import DecisionExplanationService
    from app.services.pipeline_service import PipelineService
    from app.stages.stage1.runner import build_stage1_runner
    from app.stages.stage2.runner import build_stage2_runner
    from app.stages.stage3.runner import build_stage3_runner
    from app.stages.stage4.runner import build_stage4_runner
    from tests.app_helpers import make_engine, make_session_factory

    data = ("stakeholder,feedback\n"
            "sales,We urgently need dark mode for our enterprise customers.\n"
            "engineering,Dark mode is risky and time-consuming to build across all screens.\n")
    result = ingest_csv(data, default_stakeholder=None)
    assert result.accepted_count == 2

    # FeedbackEntry from ingestion is exactly what the pipeline begins at.
    assert all(isinstance(e, FeedbackEntry) for e in result.entries)

    session = make_session_factory(make_engine())()
    try:
        client = LocalHeuristicClient()
        explanation = DecisionExplanationService(
            DecisionRepository(session), FeatureRepository(session),
            ConflictRepository(session), SignalRepository(session))
        service = PipelineService(
            session,
            stage1_runner=build_stage1_runner(client),
            stage2_runner=build_stage2_runner(client),
            stage3_runner=build_stage3_runner(client),
            stage4_runner=build_stage4_runner(client),
            explanation_service=explanation)
        run = service.analyze(list(result.entries))
        assert run.succeeded and run.failed_stage is None
        assert len(run.signals) == 2 and len(run.decisions) >= 1
    finally:
        session.close()
