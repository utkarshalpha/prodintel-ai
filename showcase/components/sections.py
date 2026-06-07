"""Showcase section renderers -- all six steps, driven by the snapshot.

Streamlit primitives only (bordered containers, expanders, columns, metrics, colored
badges). Every list is iterated in snapshot order (already deterministic) and every
section degrades gracefully when its data is absent.
"""

from __future__ import annotations

import streamlit as st

__all__ = [
    "render_metric_row",
    "render_signals",
    "render_analysis",
    "render_features",
    "render_conflicts",
    "render_decision",
    "render_why",
]

_STAKE = {
    "sales": ("green", "💼"), "engineering": ("red", "🛠️"), "customer": ("blue", "🧑‍💻"),
    "support": ("orange", "🎧"), "leadership": ("violet", "🧭"),
}
_REC = {
    "build_now": ("green", "✅"), "build_later": ("orange", "🕒"),
    "reject": ("red", "🚫"), "needs_discussion": ("violet", "💬"),
}


def _stake(s: str) -> tuple[str, str]:
    return _STAKE.get(s, ("gray", "•"))


def _rec(r: str) -> tuple[str, str]:
    return _REC.get(r, ("gray", "•"))


def _src_map(snapshot: dict) -> dict[str, str]:
    return {s["id"]: s["source_type"] for s in snapshot.get("signals", [])}


def _empty(msg: str) -> None:
    st.info(f"_{msg}_")


def _pretty(value: str) -> str:
    return value.replace("_", " ").title()


# --------------------------------------------------------------------------- #
# Metric row (rendered inside the hero)
# --------------------------------------------------------------------------- #
def render_metric_row(snapshot: dict) -> None:
    citations = sum(len(v) for v in snapshot.get("framework_citations", {}).values())
    cols = st.columns(5)
    cols[0].metric("Signals", len(snapshot.get("signals", [])))
    cols[1].metric("Features", len(snapshot.get("features", [])))
    cols[2].metric("Conflicts", len(snapshot.get("conflicts", [])))
    cols[3].metric("Decisions", len(snapshot.get("decisions", [])))
    cols[4].metric("Framework Citations", citations)


# --------------------------------------------------------------------------- #
# 1 -- Signals
# --------------------------------------------------------------------------- #
def render_signals(snapshot: dict) -> None:
    signals = snapshot.get("signals", [])
    if not signals:
        _empty("No stakeholder signals in the snapshot.")
        return
    cols = st.columns(2)
    for i, sig in enumerate(signals):
        color, emoji = _stake(sig["source_type"])
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f":{color}-background[{sig['source_type'].upper()}]  {emoji}")
                st.write(sig["raw_text"])


# --------------------------------------------------------------------------- #
# 2 -- Analysis
# --------------------------------------------------------------------------- #
def render_analysis(snapshot: dict) -> None:
    analyzed = snapshot.get("analyzed_signals", [])
    if not analyzed:
        _empty("No analyzed signals in the snapshot.")
        return
    for a in analyzed:
        color, emoji = _stake(a["stakeholder_type"])
        with st.container(border=True):
            st.markdown(f":{color}-background[{a['stakeholder_type'].upper()}]  {emoji}  **{a['intent']}**")
            m = st.columns(2)
            m[0].metric("Urgency", f"{a['urgency']}/5")
            m[1].metric("Sentiment", f"{a['sentiment']:+.2f}")
            st.caption("Extracted claims — each anchored to a character span in the original text:")
            for claim in a.get("claims", []):
                st.markdown(f"- {claim['text']}")
                st.markdown(f"> :green-background[“{claim['quoted_text']}”] — *verbatim from the signal*")


# --------------------------------------------------------------------------- #
# 3 -- Features
# --------------------------------------------------------------------------- #
def render_features(snapshot: dict) -> None:
    features = snapshot.get("features", [])
    if not features:
        _empty("No features extracted in the snapshot.")
        return
    src = _src_map(snapshot)
    for i, feat in enumerate(features):
        with st.expander(f"🧩 {feat['title']}", expanded=(i == 0)):
            st.caption(feat["jtbd"])
            links = ", ".join(f"{sid} ({src.get(sid, '?')})" for sid in feat.get("source_signal_ids", []))
            st.markdown(f"**Built from signals:** {links or '—'}")
    unassigned = snapshot.get("unassigned_signal_ids", [])
    if unassigned:
        labelled = ", ".join(f"{sid} ({src.get(sid, '?')})" for sid in unassigned)
        st.caption(f"Unassigned: {labelled} — strategic context, not a feature.")


# --------------------------------------------------------------------------- #
# 4 -- Conflicts
# --------------------------------------------------------------------------- #
def _party(col, party: dict) -> None:
    color = "green" if party["stance"] == "advocate" else "red"
    with col:
        st.markdown(f":{color}-background[{party['stance'].upper()}] — **{party['stakeholder_type']}**")
        st.write(party["summary"])
        st.caption("Evidence: " + ", ".join(party.get("evidence_signal_ids", [])))


def render_conflicts(snapshot: dict) -> None:
    conflicts = snapshot.get("conflicts", [])
    if not conflicts:
        _empty("No conflicts detected — stakeholders agree. (An empty result is a valid outcome.)")
        return
    for conf in conflicts:
        with st.container(border=True):
            st.markdown(f"**{conf['conflict_type'].upper()} conflict** · severity {conf['severity']}/5")
            parties = conf.get("parties", [])
            if len(parties) >= 2:
                left, mid, right = st.columns([6, 1, 6])
                _party(left, parties[0])
                mid.markdown("### ⟂")
                _party(right, parties[1])
            else:
                for p in parties:
                    _party(st.container(), p)


# --------------------------------------------------------------------------- #
# 5 -- Decision (hero)
# --------------------------------------------------------------------------- #
def _hero_decision(dec: dict, snapshot: dict) -> None:
    pool = {p["chunk_id"]: p for p in snapshot.get("framework_pool", [])}
    src = _src_map(snapshot)
    color, emoji = _rec(dec["recommendation"])

    with st.container(border=True):
        st.markdown(f"### {emoji} {dec['title']}")
        top = st.columns(3)
        top[0].metric("Recommendation", _pretty(dec["recommendation"]))
        top[1].metric("Priority", f"#{dec['priority_rank']}")
        top[2].metric("Confidence", f"{dec.get('confidence', {}).get('score', '—')}")

        st.markdown("**Rationale**")
        st.write(dec["rationale"])

        cols = st.columns(2)
        cols[0].markdown("**✅ Acknowledged conflicts**")
        cols[0].markdown("\n".join(f"- {cid}" for cid in dec.get("acknowledged_conflict_ids", [])) or "- —")
        cols[1].markdown("**🧾 Evidence signals**")
        cols[1].markdown(
            "\n".join(f"- {sid} ({src.get(sid, '?')})" for sid in dec.get("evidence_signal_ids", [])) or "- —"
        )

        st.divider()
        citations = snapshot.get("framework_citations", {}).get(dec["id"], [])
        st.markdown("**📚 Framework grounding** — retrieved passages this decision is grounded in")
        if citations:
            for cit in citations:
                passage = pool.get(cit["chunk_id"], {})
                with st.container(border=True):
                    st.markdown(f"**{passage.get('framework', '?')}** — {passage.get('source_title', '?')}  ·  "
                                f"retrieval score {cit['retrieval_score']}")
                    if passage.get("content"):
                        st.caption(f"“{passage['content']}”")
        else:
            st.caption("This decision grounded on no framework knowledge.")

    st.info(
        "💡 **Why framework grounding matters:** the recommendation is grounded in real PM frameworks "
        "(RICE, Kano), and the decision-integrity gate **rejects any citation that wasn't actually "
        "retrieved** — so the model cannot fabricate its sources."
    )
    st.caption("⚠ A decision that ignored a known conflict over its subject would be rejected by the "
               "decision-integrity gate and never persisted (the data includes that rejected example).")


def render_decision(snapshot: dict) -> None:
    decisions = snapshot.get("decisions", [])
    if not decisions:
        _empty("No decisions synthesized in the snapshot.")
        return
    _hero_decision(decisions[0], snapshot)  # rank 1, the hero
    others = decisions[1:]
    if others:
        st.markdown("**Other decisions**")
        for dec in others:
            color, emoji = _rec(dec["recommendation"])
            with st.container(border=True):
                st.markdown(f":{color}-background[{dec['recommendation'].upper()}] {emoji}  "
                            f"**{dec['title']}** · priority #{dec['priority_rank']}")
                st.write(dec["rationale"])


# --------------------------------------------------------------------------- #
# 6 -- Why (the real /why payload, as a Decision -> Evidence -> Source flow)
# --------------------------------------------------------------------------- #
def _flow_band(decision_title: str, evidence_count: int) -> None:
    a, ar1, b, ar2, c = st.columns([5, 1, 5, 1, 5])
    a.markdown(f"#### 🎯 Decision")
    a.caption(decision_title)
    ar1.markdown("#### →")
    b.markdown("#### 🧾 Evidence")
    b.caption(f"{evidence_count} backing signals")
    ar2.markdown("#### →")
    c.markdown("#### 💬 Source text")
    c.caption("the original stakeholder quotes")


def render_why(snapshot: dict) -> None:
    why = snapshot.get("why")
    if not why:
        _empty("No /why explanation in the snapshot.")
        return

    st.markdown("_Every claim shown below is traceable to original stakeholder text._")
    _flow_band(why["decision"]["title"], len(why.get("signals", [])))

    st.info(
        "💡 **Why provenance matters:** every recommendation is auditable back to the exact source quote — "
        "there is no black box. This is the `/why` endpoint walking the real provenance graph."
    )

    # Level 1 -- the decision and its subject.
    st.markdown(f"**Decision:** {why['decision']['title']}")
    if why.get("subject_feature"):
        st.markdown(f"**Subject feature:** {why['subject_feature']['title']}")

    # Level 2 -- how it's backed (grouped).
    cols = st.columns(2)
    cols[0].markdown("**🧾 Direct evidence**")
    cols[0].markdown("\n".join(f"- {e['signal_id']}" for e in why.get("direct_evidence", [])) or "- —")
    cols[1].markdown("**⟂ Acknowledged conflicts**")
    cols[1].markdown(
        "\n".join(f"- {c['id']} ({c['conflict_type']})" for c in why.get("conflicts", [])) or "- —"
    )

    # Level 3 -- traced to original stakeholder text (visible, not hidden behind clicks).
    st.markdown("**💬 Traced to original stakeholder text**")
    signals = why.get("signals", [])
    if not signals:
        st.caption("No referenced signals.")
    for sig in signals:
        color, emoji = _stake(sig["source_type"])
        with st.container(border=True):
            st.markdown(f":{color}-background[{sig['source_type'].upper()}]  {emoji}")
            st.caption(f"reached via {', '.join(sig.get('reached_via', []))}")
            analysis = sig.get("analysis")
            if analysis:
                for claim in analysis.get("claims", []):
                    st.markdown(f"> :green-background[“{claim['quoted_text']}”]")
            else:
                st.caption("(no analysis)")

    integrity = why.get("integrity", {})
    if integrity.get("complete"):
        st.success("✅ Provenance integrity: complete — every referenced signal resolved to a real row.")
    else:
        st.error("Provenance integrity: incomplete — unresolved: "
                 + ", ".join(integrity.get("unresolved_signal_ids", [])))
