"""LocalHeuristicClient -- a deterministic, input-derived ``ToolCallClient``.

No network, no API key. It implements the same ``complete(*, system, messages, tool)``
protocol as :class:`~app.ai_clients.claude_client.ClaudeToolClient`, dispatches on the
tool name, parses the stage prompt, and emits gate-valid Stage 1-4 contracts computed
**from the actual feedback text** -- so outputs change with the input and are identical
for identical input.

It is an **illustrative local engine, NOT equivalent to the Claude reasoning model**:
titles are keyword-derived, rationales are templated, and clustering/stance detection use
small fixed lexicons. Its value is letting the *architecture* (validation gates, provenance,
``/why``) run on a user's own input with no API key.

Gate-safety by construction (per the L3 validator audit):

* **Grounding (S1):** each claim's text is an exact, stripped substring of the raw signal
  and its span are that substring's offsets -> token overlap 1.0; only sentences with >=1
  content token (per ``grounding.normalize``) are emitted, guaranteeing >=1 grounded claim.
* **Provenance (S2):** union-find over *every* parsed signal -> a full, disjoint partition
  with ``unassigned_signal_ids = []``.
* **Conflict integrity (S3):** a conflict is emitted only for a feature backed by >=2
  *distinct* stakeholders whose stances oppose (>=1 advocate AND >=1 risk_flag/oppose); one
  position per distinct stakeholder; ``stakeholders`` == the position-stakeholder set.
* **Decision integrity (S4):** one decision per feature; acknowledge exactly the conflicts
  whose subject is that feature (all of them); evidence falls back to all input signals so
  it is never empty; framework citations are a subset of the injected pool.

A deterministic client cannot be repaired by retry feedback, so it must pass on the first
attempt; the conservative heuristics above ensure that. If a pathological input still
defeats a gate, the stage fails and ``PipelineService`` records it and returns the partial
run (no crash).
"""

from __future__ import annotations

import re

from app.ai_contracts.validation.grounding import normalize as _ground_normalize
from app.ai_runtime.interfaces import LLMToolResponse
from app.stages.stage1 import prompts as s1p
from app.stages.stage2 import prompts as s2p
from app.stages.stage3 import prompts as s3p
from app.stages.stage4 import prompts as s4p

__all__ = ["LocalHeuristicClient"]

# ---- prompt parsing -------------------------------------------------------- #
_SIGNAL_ID_RE = re.compile(r"signal_id:\s*([0-9a-fA-F-]{36})")
_RAW_RE = re.compile(r"<<<SIGNAL>>>\n(.*)\n<<<END>>>", re.DOTALL)
_HINT_RE = re.compile(r"Known source channel \(hint\):\s*(\w+)")
_SIGNAL_BLOCK_RE = re.compile(r"signal_id:\s*([0-9a-fA-F-]{36})(.*?)(?=signal_id:|\Z)", re.DOTALL)
_ROLE_RE = re.compile(r"(?:stakeholder|source):\s*(\w+)")
_FEATURE_RE = re.compile(r"feature_id:\s*([0-9a-fA-F-]{36})\s*\|\s*title:\s*([^\n]+)")
_CONFLICT_SUBJECT_RE = re.compile(r"conflict_id:\s*([0-9a-fA-F-]{36})\s*\|\s*subject_id:\s*([0-9a-fA-F-]{36})")
_CHUNK_ID_RE = re.compile(r"chunk_id:\s*([0-9a-fA-F-]{36})")
_SENT_SPLIT_RE = re.compile(r"[.!?\n]+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# ---- small fixed lexicons (post-tokenization tokens only) ------------------ #
_ADVOCACY = frozenset({
    "need", "needs", "needed", "want", "wants", "critical", "must", "important", "prioritize",
    "unblock", "approve", "support", "essential", "please", "add", "love", "great", "value",
    "valuable", "help", "helpful", "key", "crucial",
})
_RISK = frozenset({
    "risk", "risky", "concern", "concerns", "slow", "bug", "bugs", "weeks", "effort", "security",
    "hard", "complex", "difficult", "expensive", "costly", "fragile", "consuming", "blocker",
    "unstable", "break", "breaks", "broken", "regression",
})
_OPPOSE = frozenset({"reject", "oppose", "against", "unnecessary", "waste", "pointless"})
_URGENCY = frozenset({"urgent", "urgently", "critical", "blocker", "asap", "immediately", "now",
                      "must", "required", "cannot"})
_RESOURCE = frozenset({"weeks", "time", "capacity", "effort", "resource", "resources", "bandwidth",
                       "timeline", "schedule", "consuming", "months", "hours"})
_PRIORITY = frozenset({"priority", "prioritize", "first", "urgent", "order", "before", "backlog"})
_KW_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "want", "need", "would", "could", "should", "make",
    "get", "also", "really", "keep", "asking", "across", "all", "our", "are", "was", "but", "not",
    "can", "will", "from", "into", "about", "they", "them", "their", "have", "has", "had", "the",
    "you", "your", "i", "we", "it", "is", "to", "of", "a", "an", "on", "in", "so", "as", "at", "by",
    "be", "do", "if", "or", "build", "across", "screens", "team", "feature", "feedback",
})


def _word_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _count(tokens: list[str], lexicon: frozenset[str]) -> int:
    return sum(1 for t in tokens if t in lexicon)


def _keywords(text: str, top: int = 3) -> list[str]:
    """Salient content keywords, ranked by frequency then first occurrence (deterministic)."""

    toks = [t for t in _word_tokens(text) if len(t) >= 3 and t not in _KW_STOPWORDS]
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for i, t in enumerate(toks):
        counts[t] = counts.get(t, 0) + 1
        first.setdefault(t, i)
    ranked = sorted(counts, key=lambda t: (-counts[t], first[t]))
    return ranked[:top]


def _sentiment(text: str) -> float:
    toks = _word_tokens(text)
    pos = _count(toks, _ADVOCACY)
    neg = _count(toks, _RISK) + _count(toks, _OPPOSE)
    if pos + neg == 0:
        return 0.0
    return round((pos - neg) / (pos + neg), 1)


def _urgency(text: str) -> int:
    toks = _word_tokens(text)
    return max(1, min(5, 3 + _count(toks, _URGENCY) + (1 if "!" in text else 0)))


def _stance(text: str) -> str:
    toks = _word_tokens(text)
    adv, risk, opp = _count(toks, _ADVOCACY), _count(toks, _RISK), _count(toks, _OPPOSE)
    if opp > 0 and opp >= adv:
        return "oppose"
    if risk > 0 and risk >= adv:
        return "risk_flag"
    if adv > 0:
        return "advocate"
    return "neutral"


def _conflict_type(text: str) -> str:
    toks = _word_tokens(text)
    if _count(toks, _RISK):
        return "risk"
    if _count(toks, _RESOURCE):
        return "resource"
    if _count(toks, _PRIORITY):
        return "priority"
    return "strategic"


def _first_sentence(text: str) -> str:
    for part in _SENT_SPLIT_RE.split(text):
        part = part.strip()
        if part:
            return part
    return text.strip()


def _content_claims(raw: str, *, limit: int = 3) -> list[tuple[str, int, int]]:
    """Up to ``limit`` claims: each is a stripped sentence (>=1 content token) + exact span."""

    out: list[tuple[str, int, int]] = []
    for part in _SENT_SPLIT_RE.split(raw):
        text = part.strip()
        if not text or not _ground_normalize(text):
            continue
        start = raw.find(text)
        if start < 0:
            continue
        out.append((text, start, start + len(text)))
        if len(out) >= limit:
            break
    if not out:  # fallback: the whole signal (grounding decides; may degrade for empty content)
        text = raw.strip() or raw
        start = max(0, raw.find(text))
        out.append((text, start, start + len(text)))
    return out


def _signal_section(content: str) -> str:
    """The slice of a prompt containing only the signal blocks (excludes later sections)."""

    start = content.find("Stakeholder signals:")
    body = content[start:] if start >= 0 else content
    for header in ("Detected conflicts:", "Framework knowledge"):
        idx = body.find(header)
        if idx >= 0:
            body = body[:idx]
    return body


def _parse_signals(content: str) -> list[dict]:
    out = []
    for sid, block in _SIGNAL_BLOCK_RE.findall(_signal_section(content)):
        role = _ROLE_RE.search(block)
        out.append({"id": sid, "role": role.group(1) if role else "customer", "text": block})
    return out


def _parse_features(content: str) -> list[dict]:
    return [{"id": m.group(1), "title": m.group(2).strip()} for m in _FEATURE_RE.finditer(content)]


def _union_find(kw_sets: list[set[str]]) -> list[list[int]]:
    """Group signal indices into components; union two when they share any keyword."""

    parent = list(range(len(kw_sets)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    seen: dict[str, int] = {}
    for i, kws in enumerate(kw_sets):
        for kw in kws:
            if kw in seen:
                union(i, seen[kw])
            else:
                seen[kw] = i

    comps: dict[int, list[int]] = {}
    order: list[int] = []
    for i in range(len(kw_sets)):
        r = find(i)
        if r not in comps:
            comps[r] = []
            order.append(r)
        comps[r].append(i)
    return [comps[r] for r in order]


class LocalHeuristicClient:
    """Deterministic, input-derived ToolCallClient (illustrative; not Claude-quality)."""

    def __init__(self, *, model_id: str = "local-heuristic-v1") -> None:
        self._model_id = model_id

    def complete(self, *, system, messages, tool) -> LLMToolResponse:
        content = messages[0].content
        name = tool.name
        if name == s1p.STAGE1_TOOL_NAME:
            payload = self._stage1(content)
        elif name == s2p.STAGE2_TOOL_NAME:
            payload = self._stage2(content)
        elif name == s3p.STAGE3_TOOL_NAME:
            payload = self._stage3(content)
        elif name == s4p.STAGE4_TOOL_NAME:
            payload = self._stage4(content)
        else:  # pragma: no cover - defensive
            raise ValueError(f"LocalHeuristicClient: unknown tool {name!r}")
        return LLMToolResponse(
            tool_name=name, tool_input=payload, stop_reason="tool_use",
            model_id=self._model_id, input_tokens=max(1, len(content) // 4), output_tokens=64,
        )

    # ------------------------------------------------------------- Stage 1 ----
    def _stage1(self, content: str) -> dict:
        signal_id = _SIGNAL_ID_RE.search(content).group(1)
        raw = _RAW_RE.search(content).group(1)
        hint = _HINT_RE.search(content)
        stakeholder = hint.group(1) if hint else "customer"
        claims = _content_claims(raw)
        intent = (claims[0][0][:120]) or "Stakeholder feedback"
        return {
            "signal_id": signal_id,
            "intent": intent,
            "stakeholder_type": stakeholder,
            "urgency": _urgency(raw),
            "sentiment": _sentiment(raw),
            "extracted_claims": [
                {"text": t, "source_span": [s, e], "claim_confidence": 0.7} for (t, s, e) in claims
            ],
            "confidence": {"score": 0.7, "components": {"span_grounding_ratio": 1.0}},
        }

    # ------------------------------------------------------------- Stage 2 ----
    def _stage2(self, content: str) -> dict:
        signals = _parse_signals(content)
        kw_sets = [set(_keywords(s["text"], top=3)) for s in signals]
        features = []
        for i, comp in enumerate(_union_find(kw_sets), start=1):
            members = [signals[j] for j in comp]
            combined = " ".join(m["text"] for m in members)
            kws = _keywords(combined, top=2)
            title = (" ".join(kws).title()[:120]) if kws else "Feature"
            top_kw = kws[0] if kws else "this request"
            features.append({
                "feature_id": f"f{i}",
                "title": title or "Feature",
                "description": f"Clustered from {len(members)} signal(s) regarding {top_kw}.",
                "jtbd": f"When {top_kw} comes up, I want it addressed, so the team can act on it.",
                "source_signal_ids": [m["id"] for m in members],
                "confidence": {"score": 0.6, "components": {"cluster_cohesion": round(min(1.0, 0.5 + 0.1 * len(members)), 2)}},
            })
        return {"features": features, "unassigned_signal_ids": []}

    # ------------------------------------------------------------- Stage 3 ----
    def _stage3(self, content: str) -> dict:
        features = _parse_features(content)
        signals = _parse_signals(content)
        conflicts = []
        ci = 1
        for feat in features:
            fk = set(_keywords(feat["title"], top=3))
            if not fk:
                continue
            assoc = [s for s in signals if fk & set(_keywords(s["text"], top=3))]
            by_role: dict[str, dict] = {}
            for s in assoc:
                slot = by_role.setdefault(s["role"], {"text": [], "ids": []})
                slot["text"].append(s["text"])
                slot["ids"].append(s["id"])
            positions = []
            for role in sorted(by_role):
                combined = " ".join(by_role[role]["text"])
                stance = _stance(combined)
                if stance == "neutral":
                    continue
                positions.append({
                    "stakeholder": role, "stance": stance,
                    "summary": (_first_sentence(combined) or f"{role} position")[:200],
                    "evidence_signal_ids": sorted(set(by_role[role]["ids"])),
                })
            stances = {p["stance"] for p in positions}
            if len(positions) < 2 or "advocate" not in stances or not (stances & {"risk_flag", "oppose"}):
                continue
            conflicts.append({
                "conflict_id": f"c{ci}",
                "conflict_type": _conflict_type(" ".join(s["text"] for s in assoc)),
                "severity": min(5, 2 + len(positions)),
                "subject_type": "feature",
                "subject_id": feat["id"],
                "stakeholders": [p["stakeholder"] for p in positions],
                "positions": positions,
                "evidence_signal_ids": sorted({i for p in positions for i in p["evidence_signal_ids"]}),
                "confidence": {"score": 0.65, "components": {"opposition_strength": round(min(1.0, 0.4 + 0.2 * len(positions)), 2)}},
            })
            ci += 1
        return {"conflicts": conflicts}

    # ------------------------------------------------------------- Stage 4 ----
    def _stage4(self, content: str) -> dict:
        features = _parse_features(content)
        signals = _parse_signals(content)
        conflict_pairs = _CONFLICT_SUBJECT_RE.findall(content)
        chunk_ids = _CHUNK_ID_RE.findall(content)
        all_ids = sorted({s["id"] for s in signals})

        info = []
        for feat in features:
            fk = set(_keywords(feat["title"], top=3))
            assoc = [s for s in signals if fk & set(_keywords(s["text"], top=3))]
            ev = sorted({s["id"] for s in assoc}) or all_ids  # D1: never empty
            ack = [cid for cid, subj in conflict_pairs if subj == feat["id"]]
            combined = " ".join(s["text"] for s in (assoc or signals))
            score = sum(_urgency(s["text"]) + _sentiment(s["text"]) for s in assoc)
            info.append({"feat": feat, "ev": ev, "ack": ack, "combined": combined, "score": score,
                         "stakeholders": sorted({s["role"] for s in assoc})})

        ranked = sorted(range(len(info)), key=lambda i: (-info[i]["score"], info[i]["feat"]["id"]))
        rank_of = {i: pos + 1 for pos, i in enumerate(ranked)}
        top_idx = ranked[0] if ranked else None

        decisions = []
        for i, it in enumerate(info):
            feat, ev, ack = it["feat"], it["ev"], it["ack"]
            if ack:
                rec, verb = "needs_discussion", "Discuss"
            elif _stance(it["combined"]) == "advocate" and _urgency(it["combined"]) >= 4:
                rec, verb = "build_now", "Prioritize"
            else:
                rec, verb = "build_later", "Plan"
            fw = list(chunk_ids) if (chunk_ids and i == top_idx) else []
            sh = ", ".join(it["stakeholders"]) or "stakeholders"
            rationale = (
                f"Backed by {len(ev)} signal(s) from {sh}. "
                f"{f'{len(ack)} conflict(s) to resolve. ' if ack else 'No blocking conflicts. '}"
                f"{'Grounded in retrieved framework knowledge. ' if fw else 'No framework grounding applied. '}"
                f"Heuristic recommendation: {rec.replace('_', ' ')}."
            )
            decisions.append({
                "decision_id": f"d{i + 1}", "subject_type": "feature", "subject_id": feat["id"],
                "recommendation": rec, "title": f"{verb} {feat['title']}"[:160], "rationale": rationale,
                "priority_rank": rank_of[i],
                "acknowledged_conflict_ids": ack,
                "evidence_signal_ids": ev,
                "framework_citation_ids": fw,
                "confidence": {"score": 0.6, "components": {"evidence_coverage": round(min(1.0, 0.3 + 0.1 * len(ev)), 2)}},
            })
        return {"decisions": decisions}
