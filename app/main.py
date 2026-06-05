"""ASGI entrypoint for production.

Run with: ``uvicorn app.main:app``. Requires ``ANTHROPIC_API_KEY`` in the
environment (and optionally ``ANTHROPIC_MODEL`` etc.). The app is built by composing
the existing FastAPI app with a live Claude-backed Stage 1 runner.
"""

from __future__ import annotations

from app.ai_clients.wiring import create_production_app

app = create_production_app()
