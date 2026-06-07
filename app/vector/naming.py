"""Deterministic Chroma collection-name derivation (Phase 6C-3A).

A Chroma collection is fixed-dimension, and vectors from different embedding models
or dimensionalities are mutually incomparable, so the corpus uses **one collection
per (embedding_model_id, dim)** pairing. The name is derived deterministically from
those two values and a configurable prefix; ``corpus_version`` is NOT in the name (it
lives in vector metadata and is filtered at query time).

The derived name always satisfies Chroma's collection-name constraints: 3-63
characters, only ``[a-z0-9_]`` (a subset of Chroma's allowed set), starts and ends
with an alphanumeric, no consecutive dots, and never a valid IPv4 address. An 8-hex
suffix hashing the *full* ``(model_id, dim)`` guarantees that two model ids which
slugify to the same prefix (or are truncated) still map to distinct collections.

Pure stdlib -- imports no chromadb, no config; safe to import anywhere.
"""

from __future__ import annotations

import re
from hashlib import sha256

__all__ = ["build_collection_name", "MAX_COLLECTION_NAME_LENGTH"]

#: Chroma's maximum collection-name length.
MAX_COLLECTION_NAME_LENGTH = 63
_HASH_LENGTH = 8
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    """Lowercase, collapse non-alphanumeric runs to ``_``, and strip edge ``_``."""

    return _NON_ALNUM.sub("_", value.lower()).strip("_")


def build_collection_name(prefix: str, embedding_model_id: str, dim: int) -> str:
    """Derive a deterministic, Chroma-compliant collection name.

    Form: ``{prefix}_{model_slug}_{dim}_{hash8}``. ``model_slug`` is truncated as
    needed to keep the whole name within :data:`MAX_COLLECTION_NAME_LENGTH`; the hash
    of the full ``(embedding_model_id, dim)`` preserves collision resistance even when
    the slug is truncated or two ids slugify identically.
    """

    if dim < 1:
        raise ValueError("dim must be >= 1")

    base = _slug(prefix) or "col"
    digest = sha256(f"{embedding_model_id}\x00{dim}".encode("utf-8")).hexdigest()[:_HASH_LENGTH]
    suffix = f"{dim}_{digest}"  # always ascii [0-9a-f_], ends with a hex (alphanumeric)
    model_slug = _slug(embedding_model_id) or "model"

    # Trim a pathologically long prefix first, keeping room for model + suffix.
    max_base = MAX_COLLECTION_NAME_LENGTH - (len("_") + 1 + len("_") + len(suffix))
    if len(base) > max_base:
        base = base[:max_base].strip("_") or "col"

    # Budget the remaining space for the model slug: base + "_" + model + "_" + suffix.
    budget = MAX_COLLECTION_NAME_LENGTH - (len(base) + 1 + 1 + len(suffix))
    if budget < len(model_slug):
        model_slug = model_slug[: max(budget, 1)].strip("_") or "m"

    return f"{base}_{model_slug}_{suffix}"
