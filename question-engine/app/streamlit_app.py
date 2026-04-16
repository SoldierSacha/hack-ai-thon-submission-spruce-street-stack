from __future__ import annotations
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
from datetime import date
import numpy as np
import plotly.graph_objects as go
from src.db import Repo
from src.llm import LlmClient
from src.taxonomy import load_taxonomy, SCHEMA_DESCRIPTION_FIELDS
from src.ranker import build_field_cluster_map, rank_fields
from src.flow import AskFlow
from src.models import SUB_RATING_KEYS, FieldState, ScoredField, SubmitResult

try:
    from streamlit_mic_recorder import speech_to_text
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

# ---------- page config ----------
st.set_page_config(page_title="Ask What Matters", layout="wide", page_icon="\U0001f50d")

# ---------- resources (cached) ----------
@st.cache_resource
def get_repo():
    return Repo(REPO_ROOT / "data" / "state.sqlite")

@st.cache_resource
def get_llm():
    return LlmClient(cache_dir=REPO_ROOT / "data" / "cache")

@st.cache_resource
def get_taxonomy():
    return load_taxonomy(REPO_ROOT / "config" / "taxonomy.yaml")

@st.cache_resource
def get_flow():
    repo = get_repo()
    llm = get_llm()
    topics = get_taxonomy()
    return AskFlow(
        repo=repo, llm=llm,
        taxonomy=topics,
        topic_embeddings={},
        field_cluster=build_field_cluster_map(topics),
        weights_path=str(REPO_ROOT / "config" / "weights.yaml"),
    )


def _max_review_date(repo):
    """Find the newest acquisition_date across all reviews (reproducible 'today')."""
    all_dates = []
    for p in repo.list_properties():
        for r in repo.list_reviews_for(p.eg_property_id):
            all_dates.append(r.acquisition_date)
    return max(all_dates) if all_dates else date.today()


# ---------- CSS ----------
st.markdown("""
<style>
/* Header bar */
.header-bar {
    background: linear-gradient(135deg, #00355F 0%, #004E8C 100%);
    color: #ffffff;
    padding: 20px 28px 16px 28px;
    border-radius: 0 0 16px 16px;
    margin: -1rem -1rem 1.2rem -1rem;
}
.header-bar h1 {
    margin: 0 0 2px 0;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.3px;
}
.header-bar p {
    margin: 0;
    font-size: 14px;
    opacity: 0.82;
}

/* Property card */
.prop-card {
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07), 0 2px 12px rgba(0,0,0,0.04);
    padding: 20px;
    margin-bottom: 16px;
}
.prop-card h3 { margin: 0 0 8px 0; font-size: 18px; color: #1e293b !important; }
.prop-card, .prop-card * { color: #1e293b; }
.prop-meta { color: #64748b !important; font-size: 13px; margin-top: 4px; }

/* Star pill */
.star-pill {
    display: inline-block;
    background: #FBCE38;
    color: #00355F;
    font-weight: 700;
    font-size: 13px;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 6px;
}

/* Field rows */
.field-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 4px;
    border-bottom: 1px solid #f1f5f9;
    font-size: 13.5px;
}
.field-row:last-child { border-bottom: none; }

/* Status pills */
.pill {
    display: inline-block;
    font-size: 12px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 12px;
    white-space: nowrap;
}
.pill-known   { background: #dcfce7; color: #16a34a; }
.pill-stale   { background: #fef3c7; color: #d97706; }
.pill-drifting { background: #fef3c7; color: #d97706; }
.pill-missing { background: #fee2e2; color: #ef4444; }

/* Flash animation */
.flash { animation: flash 1.8s ease-out; background: transparent; }
@keyframes flash { 0% { background: #fef08a; } 100% { background: transparent; } }

/* Mini score bar */
.score-bar-wrap {
    display: inline-flex;
    align-items: center;
    margin-left: 8px;
    vertical-align: middle;
}
.score-bar {
    display: flex;
    width: 80px;
    height: 5px;
    border-radius: 3px;
    overflow: hidden;
    background: #f1f5f9;
}
.seg-m { background: #ef4444; }
.seg-s { background: #d97706; }
.seg-c { background: #6366f1; }
.score-bar-wrap {
    position: relative;
}
.score-bar-wrap .score-tip {
    position: absolute;
    bottom: 14px;
    left: 50%;
    transform: translateX(-50%);
    background: #1e293b;
    color: #fff;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 6px;
    white-space: nowrap;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.1s;
    z-index: 100;
}
.score-bar-wrap:hover .score-tip {
    opacity: 1;
}

/* Question card */
.q-card {
    padding: 18px 20px;
    border: 2px solid #6366f1;
    border-radius: 12px;
    margin: 14px 0 10px 0;
    background: #fafaff;
}
.q-card, .q-card * { color: #1e293b; }
.q-card .q-text { font-size: 17px; font-weight: 600; color: #1e293b !important; margin-bottom: 6px; }
.q-card .q-reason { color: #6b7280 !important; font-size: 13px; font-style: italic; }

/* Score detail */
.score-detail {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 12px;
    color: #475569;
    line-height: 1.7;
    padding: 8px 0;
}
.score-detail .val { color: #6366f1; font-weight: 600; }

/* Rank & cluster badges */
.rank-badge {
    display: inline-block;
    background: #6366f1;
    color: #ffffff;
    font-size: 12px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 12px;
    margin-right: 6px;
}
.cluster-badge {
    display: inline-block;
    border: 1.5px solid #a5b4fc;
    color: #6366f1;
    font-size: 12px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 12px;
    background: #eef2ff;
}

/* Enrichment strip */
.enrich-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0;
}
.enrich-chip {
    display: inline-flex;
    align-items: center;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 16px;
    gap: 4px;
}
.enrich-lang    { background: #dcfce7; color: #16a34a; }
.enrich-trans   { background: #dbeafe; color: #2563eb; }
.enrich-embed   { background: #ede9fe; color: #7c3aed; }
.enrich-topics  { background: #fef3c7; color: #d97706; }

/* Mentioned-topic chips */
.topic-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 16px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    margin: 2px;
    color: #334155;
}
.topic-tag .sent-pos { color: #16a34a; font-weight: 700; }
.topic-tag .sent-neg { color: #ef4444; font-weight: 700; }
.topic-tag .sent-neu { color: #94a3b8; }
.topic-tag .assertion { font-style: italic; color: #64748b; font-weight: 400; }

/* Section header */
.section-hdr {
    color: #00355F;
    text-transform: uppercase;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.2px;
    margin: 18px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 2px solid #e2e8f0;
}

/* Coverage label */
.coverage-label {
    font-size: 13px;
    color: #64748b;
    text-align: center;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)


# ---------- cluster labels ----------
CLUSTER_LABELS = {
    "connectivity": "Connectivity",
    "room": "Room",
    "building": "Building",
    "food": "Food & Dining",
    "amenities": "Amenities",
    "service": "Service",
    "location": "Location",
    "parking": "Parking & Access",
    "policies": "Policies",
    "value": "Value",
    "meta": "Accuracy & Safety",
}

_RADAR_CLUSTERS = [
    "connectivity", "room", "building", "food", "amenities",
    "service", "location", "parking", "policies", "value", "meta",
]


# ---------- chart helpers ----------
def make_radar_chart(
    field_states: list[FieldState],
    topics: list,
    property_id: str,
) -> go.Figure:
    """Scatterpolar showing % known fields per cluster."""
    cluster_map = build_field_cluster_map(topics)
    totals: dict[str, int] = {c: 0 for c in _RADAR_CLUSTERS}
    knowns: dict[str, int] = {c: 0 for c in _RADAR_CLUSTERS}

    for fs in field_states:
        if fs.eg_property_id != property_id:
            continue
        cl = cluster_map.get(fs.field_id, "")
        if cl not in totals:
            continue
        totals[cl] += 1
        if fs.value_known:
            knowns[cl] += 1

    labels = [CLUSTER_LABELS.get(c, c.title()) for c in _RADAR_CLUSTERS]
    values = [
        round(100 * knowns[c] / totals[c]) if totals[c] > 0 else 0
        for c in _RADAR_CLUSTERS
    ]
    # Close the polygon
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(99,102,241,0.15)",
        line=dict(color="#6366f1", width=2),
        marker=dict(size=5, color="#6366f1"),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100], ticksuffix="%",
                tickfont=dict(size=10, color="#94a3b8"),
                gridcolor="#e2e8f0",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#475569"),
                gridcolor="#e2e8f0",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        height=320,
        margin=dict(l=50, r=50, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_donut_chart(known: int, total: int) -> go.Figure:
    """Donut chart showing overall coverage %."""
    pct = round(100 * known / total) if total > 0 else 0
    unknown = total - known
    fig = go.Figure(data=[go.Pie(
        values=[known, unknown],
        labels=["Known", "Unknown"],
        hole=0.75,
        marker=dict(colors=["#16a34a", "#e5e7eb"]),
        textinfo="none",
        hoverinfo="label+value",
    )])
    fig.update_layout(
        showlegend=False,
        height=180,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"<b>{pct}%</b>",
            x=0.5, y=0.5, font=dict(size=28, color="#00355F"),
            showarrow=False,
        )],
    )
    return fig


# ---------- helper functions ----------
def field_status(fs: FieldState | None, today: date) -> tuple[str, str]:
    """Return (label, css_class) for a field state."""
    if not fs or not fs.value_known:
        return ("Missing", "pill-missing")
    # Check EMA drift
    if (
        fs.short_ema is not None
        and fs.long_ema is not None
        and abs(fs.short_ema - fs.long_ema) >= 0.3
    ):
        return ("Drifting", "pill-drifting")
    # Check staleness (>90 days)
    if fs.last_confirmed_date:
        age = (today - fs.last_confirmed_date).days
        if age > 90:
            return ("Stale", "pill-stale")
    return ("Known", "pill-known")


def score_bar_html(sf: ScoredField) -> str:
    """Return HTML for a tiny stacked score bar with a hover tooltip."""
    total = sf.missing + sf.stale + sf.coverage + 0.001
    m_pct = sf.missing / total * 100
    s_pct = sf.stale / total * 100
    c_pct = sf.coverage / total * 100
    tip = f"No data yet {sf.missing:.0%} · Outdated {sf.stale:.0%} · Rarely discussed {sf.coverage:.0%}"
    return (
        f"<span class='score-bar-wrap'>"
        f"<span class='score-tip'>{tip}</span>"
        f"<span class='score-bar'>"
        f"<span class='seg-m' style='width:{m_pct:.0f}%'></span>"
        f"<span class='seg-s' style='width:{s_pct:.0f}%'></span>"
        f"<span class='seg-c' style='width:{c_pct:.0f}%'></span>"
        f"</span></span>"
    )


def _topic_display(fs: FieldState | None) -> tuple[str, str]:
    """Return (display_text, css_class) for a topic field state."""
    if not fs or not fs.value_known:
        return ("-- no mentions", "pill-missing")
    ema = fs.short_ema
    count = fs.mention_count
    if ema is not None:
        if ema > 0.3:
            sentiment = "Positive"
        elif ema < -0.3:
            sentiment = "Negative"
        else:
            sentiment = "Mixed"
    else:
        sentiment = "Mentioned"
    return (f"{sentiment} ({count})", "pill-known")


def _cluster_stats(cluster: str, sf_lookup: dict[str, ScoredField], field_cluster: dict[str, str]) -> tuple[int, int]:
    """Return (known_count, total_count) for fields in a cluster."""
    total = known = 0
    for fid, cl in field_cluster.items():
        if cl == cluster:
            total += 1
            sf = sf_lookup.get(fid)
            if sf and sf.field_state.value_known:
                known += 1
    return known, total


def _field_label(fid: str) -> str:
    """Human-readable label for a field_id."""
    return fid.split(":", 1)[1].replace("_", " ") if ":" in fid else fid


def _rich_why(
    sf: ScoredField,
    display_slot: int,
    all_questions: list,
    sf_lookup: dict[str, ScoredField],
    field_cluster: dict[str, str],
) -> str:
    """Build a plain-English explanation for why this question was picked."""
    cluster = sf.cluster or ""
    cluster_label = CLUSTER_LABELS.get(cluster, cluster.replace("_", " ").title())
    known, total = _cluster_stats(cluster, sf_lookup, field_cluster)
    unknown = total - known
    key = _field_label(sf.field_state.field_id)

    parts = []

    # Field-level reason
    if sf.missing > 0.5:
        parts.append(f"This property has <b>no {key} data</b> yet.")
    elif sf.stale > 0.3:
        parts.append(f"The <b>{key}</b> information looks <b>outdated</b> \u2014 recent reviews suggest it may have changed.")
    elif sf.coverage > 0.5:
        parts.append(f"Very <b>few guests</b> have commented on <b>{key}</b>.")
    else:
        parts.append(f"The <b>{key}</b> field has a moderate information gap.")

    # Cluster-level context
    if unknown > 0:
        parts.append(
            f"We\u2019re asking about <b>{cluster_label}</b> because "
            f"{unknown} of {total} field{'s' if unknown != 1 else ''} in this category "
            f"{'are' if unknown != 1 else 'is'} still unknown."
        )

    # Q2: explain why a different cluster was chosen
    if display_slot == 1 and len(all_questions) >= 2:
        q1_fid = all_questions[0][0].field_id
        q1_sf = sf_lookup.get(q1_fid)
        if q1_sf:
            q1_label = CLUSTER_LABELS.get(q1_sf.cluster or "", q1_sf.cluster or "").title()
            if q1_label != cluster_label:
                parts.append(
                    f"We chose a different category from question 1 (<b>{q1_label}</b>) "
                    f"to <b>maximize new information</b>."
                )

    # Redundancy note
    if sf.redundancy > 0.3:
        parts.append("Your review already touched on related topics, so this was slightly de-prioritized.")

    return " ".join(parts)


def _component_bars_html(sf: ScoredField) -> str:
    """Render horizontal bars for each score component."""
    components = [
        ("No data yet", sf.missing, "#ef4444"),
        ("Outdated info", sf.stale, "#d97706"),
        ("Rarely discussed", sf.coverage, "#6366f1"),
        ("Redundancy penalty", sf.redundancy, "#94a3b8"),
    ]
    rows = []
    for label, val, color in components:
        pct = min(val * 100, 100)
        rows.append(
            f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px;'>"
            f"<span style='width:120px;color:#64748b;'>{label}</span>"
            f"<span style='flex:1;background:#f1f5f9;border-radius:3px;height:8px;overflow:hidden;'>"
            f"<span style='display:block;width:{pct:.0f}%;height:100%;background:{color};border-radius:3px;'></span>"
            f"</span>"
            f"<span style='width:36px;text-align:right;color:#475569;font-weight:600;font-size:11px;'>{val:.2f}</span>"
            f"</div>"
        )
    return "".join(rows)


def _runner_ups_html(sf: ScoredField, result: SubmitResult) -> str:
    """Show what fields ranked #2 and #3 (the ones NOT picked)."""
    # Find fields ranked just below this one that weren't picked as questions
    picked_fids = {q.field_id for q, _ in st.session_state.get("pending_questions", [])}
    runners = []
    for other in result.scored_fields:
        if other.field_state.field_id in picked_fids:
            continue
        if other.rank > sf.rank:
            runners.append(other)
        if len(runners) >= 2:
            break
    if not runners:
        return ""
    items = ", ".join(
        f"<b>#{r.rank}</b> {_field_label(r.field_state.field_id)} ({r.composite:.3f})"
        for r in runners
    )
    return (
        f"<div style='font-size:12px;color:#64748b;margin-top:6px;'>"
        f"Runner-ups not selected: {items}"
        f"</div>"
    )


def _review_overlap_html(result: SubmitResult, topics_list) -> str:
    """Show which topics the review mentioned and were de-prioritized."""
    if not result.tags:
        return ""
    topic_label_map = {t.topic_id: t.label for t in topics_list}
    mentioned = [
        topic_label_map.get(tid, tid)
        for tid, tag in result.tags.items()
        if tag.mentioned
    ]
    if not mentioned:
        return ""
    labels = ", ".join(f"<b>{m}</b>" for m in mentioned[:8])
    return (
        f"<div style='font-size:12px;color:#64748b;margin-top:6px;'>"
        f"Your review mentioned: {labels}. "
        f"These topics received a redundancy penalty so we ask about something new."
        f"</div>"
    )


def _impact_preview_html(sf: ScoredField, sf_lookup: dict[str, ScoredField]) -> str:
    """Show what answering this question will do for coverage."""
    total = len(sf_lookup)
    known = sum(1 for s in sf_lookup.values() if s.field_state.value_known)
    if total == 0:
        return ""
    pct_now = round(100 * known / total)
    pct_after = round(100 * (known + 1) / total) if not sf.field_state.value_known else pct_now
    if pct_after == pct_now:
        return ""
    return (
        f"<div style='font-size:12px;color:#16a34a;font-weight:600;margin-top:6px;'>"
        f"Answering this raises coverage from {pct_now}% \u2192 {pct_after}%"
        f"</div>"
    )


# ---------- state init ----------
def _init_state():
    ss = st.session_state
    ss.setdefault("review_text", "")
    ss.setdefault("pending_questions", [])
    ss.setdefault("submit_result", None)
    ss.setdefault("answered_fields", set())
    ss.setdefault("last_flashed_field", None)
    ss.setdefault("answer_widgets", {})
    ss.setdefault("active_property_id", None)

_init_state()

# ---------- resources ----------
repo = get_repo()
flow = get_flow()
topics = get_taxonomy()

# Group topics by cluster
_topic_clusters: dict[str, list] = {}
for _t in topics:
    _topic_clusters.setdefault(_t.cluster_id, []).append(_t)

properties = repo.list_properties()
_property_id_list = [p.eg_property_id for p in properties]
_property_label_cache: dict[str, str] = {}


def _format_property(pid: str) -> str:
    if pid not in _property_label_cache:
        p = repo.get_property(pid)
        n = len(repo.list_reviews_for(pid))
        all_fs = repo.list_field_states_for(pid)
        known = sum(1 for fs in all_fs if fs.value_known)
        total = len(all_fs)
        cov_pct = round(100 * known / total) if total > 0 else 0
        _property_label_cache[pid] = (
            f"{p.city or '?'}, {p.country or '?'} "
            f"({p.star_rating or '?'}\u2605, {n} reviews, {cov_pct}% coverage)"
        )
    return _property_label_cache[pid]


# ---------- header ----------
st.markdown(
    "<div class='header-bar'>"
    "<h1>Ask What Matters</h1>"
    "<p>Adaptive follow-up questions that fill missing property information</p>"
    "</div>",
    unsafe_allow_html=True,
)

# Property picker lock
_has_unanswered = any(not a for _, a in st.session_state.get("pending_questions", []))
_active = st.session_state.get("active_property_id")
if _has_unanswered and _active and _active in _property_id_list:
    st.session_state.property_picker = _active

picker_cols = st.columns([5, 1])
with picker_cols[0]:
    property_id = st.selectbox(
        "Property",
        _property_id_list,
        format_func=_format_property,
        key="property_picker",
        disabled=_has_unanswered,
        label_visibility="collapsed",
    )
with picker_cols[1]:
    if st.button("Reset", help="Clear session answers (DB untouched)", width="stretch"):
        for k in [
            "pending_questions", "answered_fields", "last_flashed_field",
            "review_text", "active_property_id", "submit_result",
        ]:
            st.session_state.pop(k, None)
        _init_state()
        st.rerun()

prop = repo.get_property(property_id)
today = _max_review_date(repo)

# ---------- ScoredField lookup ----------
# Use SubmitResult if available (has redundancy from the review), otherwise
# score all fields on load so the mini bars always appear.
result: SubmitResult | None = st.session_state.get("submit_result")
sf_lookup: dict[str, ScoredField] = {}
if result and result.scored_fields:
    for sf in result.scored_fields:
        sf_lookup[sf.field_state.field_id] = sf
else:
    _all_fs = repo.list_field_states_for(property_id)
    if _all_fs:
        _ranked = rank_fields(
            property_id=property_id,
            field_states=_all_fs,
            today=today,
            topic_embeddings={},
            review_embedding=None,
            field_cluster=build_field_cluster_map(topics),
            weights_path=str(REPO_ROOT / "config" / "weights.yaml"),
        )
        for sf in _ranked:
            sf_lookup[sf.field_state.field_id] = sf

# ---------- two-column layout ----------
left, right = st.columns([2, 3], gap="large")

# ========================
# LEFT COLUMN — Knowledge Panel
# ========================
with left:
    # Property card
    star_html = f"<span class='star-pill'>{prop.star_rating}\u2605</span>" if prop.star_rating else ""
    avg_html = f"{prop.guestrating_avg_expedia:.1f} avg" if prop and prop.guestrating_avg_expedia else ""
    checkin_html = ""
    if prop and prop.check_in_start_time:
        checkin_html = (
            f"<div class='prop-meta'>Check-in {prop.check_in_start_time}"
            f" \u2013 {prop.check_in_end_time or '?'}"
            f" &middot; Check-out {prop.check_out_time or '?'}</div>"
        )

    st.markdown(
        f"<div class='prop-card'>"
        f"<h3>{prop.city or '?'}, {prop.country or '?'}</h3>"
        f"{star_html} <span style='color:#64748b; font-size:13px;'>{avg_html}</span>"
        f"{checkin_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Coverage donut + Radar side by side
    all_fs = repo.list_field_states_for(property_id)
    known = sum(1 for fs in all_fs if fs.value_known)
    total = len(all_fs)

    chart_left, chart_right = st.columns([1, 2])
    with chart_left:
        st.plotly_chart(make_donut_chart(known, total), width="stretch", key="donut")
        st.markdown(
            f"<div class='coverage-label'>{known} / {total} fields known</div>",
            unsafe_allow_html=True,
        )
    with chart_right:
        st.plotly_chart(
            make_radar_chart(all_fs, topics, property_id),
            width="stretch",
            key="radar",
        )

    # Score bar legend + Sub-ratings section
    st.markdown(
        "<div style='display:flex;gap:14px;font-size:11px;color:#64748b;margin:12px 0 4px 0;'>"
        "<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
        "background:#ef4444;margin-right:3px;vertical-align:middle;'></span>No data yet</span>"
        "<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
        "background:#d97706;margin-right:3px;vertical-align:middle;'></span>Outdated</span>"
        "<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
        "background:#6366f1;margin-right:3px;vertical-align:middle;'></span>Rarely discussed</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-hdr'>Sub-ratings</div>", unsafe_allow_html=True)
    flash_field = st.session_state.last_flashed_field
    for key in SUB_RATING_KEYS:
        fid = f"rating:{key}"
        fs = repo.get_field_state(property_id, fid)
        pretty = key.replace("_", " ").title()
        css_class = "flash" if flash_field == fid else ""
        label, pill_class = field_status(fs, today)

        sf_match = sf_lookup.get(fid)
        bar_html = score_bar_html(sf_match) if sf_match else ""

        if fs and fs.value_known:
            ema = f"{fs.short_ema:.1f}" if fs.short_ema is not None else "?"
            st.markdown(
                f"<div class='field-row {css_class}'>"
                f"<span>{pretty}</span>"
                f"<span><span class='pill {pill_class}'>{label} ({ema}/5)</span>{bar_html}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='field-row {css_class}'>"
                f"<span>{pretty}</span>"
                f"<span><span class='pill {pill_class}'>{label}</span>{bar_html}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Description fields section
    st.markdown("<div class='section-hdr'>Description fields</div>", unsafe_allow_html=True)
    for sf_name in SCHEMA_DESCRIPTION_FIELDS:
        fid = f"schema:{sf_name}"
        fs = repo.get_field_state(property_id, fid)
        pretty = sf_name.replace("property_amenity_", "").replace("_", " ").title()
        css_class = "flash" if flash_field == fid else ""
        label, pill_class = field_status(fs, today)

        sf_match = sf_lookup.get(fid)
        bar_html = score_bar_html(sf_match) if sf_match else ""

        st.markdown(
            f"<div class='field-row {css_class}'>"
            f"<span>{pretty}</span>"
            f"<span><span class='pill {pill_class}'>{label}</span>{bar_html}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Guest topics section
    st.markdown("<div class='section-hdr'>Guest topics</div>", unsafe_allow_html=True)
    for cluster_id, cluster_topics in _topic_clusters.items():
        cluster_states = []
        has_known = False
        for t in cluster_topics:
            fid = f"topic:{t.topic_id}"
            fs = repo.get_field_state(property_id, fid)
            cluster_states.append((t, fs))
            if fs and fs.value_known:
                has_known = True

        known_count = sum(1 for _, fs in cluster_states if fs and fs.value_known)
        total_count = len(cluster_states)
        cluster_label = CLUSTER_LABELS.get(cluster_id, cluster_id.replace("_", " ").title())

        with st.expander(f"{cluster_label} ({known_count}/{total_count})", expanded=has_known):
            for t, fs in cluster_states:
                fid = f"topic:{t.topic_id}"
                css_class = "flash" if flash_field == fid else ""
                display_text, value_class = _topic_display(fs)

                sf_match = sf_lookup.get(fid)
                bar_html = score_bar_html(sf_match) if sf_match else ""

                st.markdown(
                    f"<div class='field-row {css_class}'>"
                    f"<span>{t.label}</span>"
                    f"<span><span class='pill {value_class}'>{display_text}</span>{bar_html}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ========================
# RIGHT COLUMN — Review & Question Flow
# ========================
with right:
    st.markdown("<div class='section-hdr'>Write your review</div>", unsafe_allow_html=True)

    # Voice input
    if MIC_AVAILABLE:
        voice_text = speech_to_text(
            language="en",
            start_prompt="\U0001f3a4 Record review",
            stop_prompt="\u23f9 Stop",
            just_once=True,
            use_container_width=False,
            key="voice_review",
        )
        if voice_text:
            st.session_state.review_text = voice_text
            st.session_state.review_input = voice_text
    else:
        st.caption("(Install `streamlit-mic-recorder` for voice input.)")

    review_text = st.text_area(
        "Your review (or leave blank for cold-start demo)",
        key="review_input",
        height=120,
        label_visibility="collapsed",
        placeholder="Write your review here, or leave blank to see cold-start gap analysis...",
    )

    if st.button("Submit review", type="primary", key="submit_review", width="stretch"):
        with st.spinner("Analyzing review \u2014 detecting language, embedding, tagging topics, scoring gaps..."):
            result = flow.submit_review(property_id, review_text, today=today)
        st.session_state.pending_questions = [(q, False) for q in result.questions]
        st.session_state.submit_result = result
        st.session_state.review_text = review_text
        st.session_state.active_property_id = property_id
        st.session_state.pending_review_id = flow._pending_review_id.get(property_id)
        st.rerun()

    # Enrichment strip (shows after submit)
    if result and st.session_state.get("active_property_id") == property_id:
        enr = result.enrichment
        st.markdown(
            f"<div class='enrich-strip'>"
            f"<span class='enrich-chip enrich-lang'>Lang: {enr.lang}</span>"
            f"<span class='enrich-chip enrich-trans'>{'Translated' if enr.translated else 'Native EN'}</span>"
            f"<span class='enrich-chip enrich-embed'>Embed: {enr.embedding_dim}d</span>"
            f"<span class='enrich-chip enrich-topics'>Topics: {enr.topics_tagged}/{enr.topics_total}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Show which topics the review mentioned
        if result.tags:
            topic_label_map = {t.topic_id: t.label for t in topics}
            mentioned = {tid: tag for tid, tag in result.tags.items() if tag.mentioned}
            if mentioned:
                chips = []
                for tid, tag in mentioned.items():
                    label = topic_label_map.get(tid, tid)
                    if tag.sentiment == 1:
                        icon, css = "\u2191", "sent-pos"
                    elif tag.sentiment == -1:
                        icon, css = "\u2193", "sent-neg"
                    else:
                        icon, css = "\u2022", "sent-neu"
                    assertion = (
                        f" <span class='assertion'>\u2014 {tag.assertion}</span>"
                        if tag.assertion else ""
                    )
                    chips.append(
                        f"<span class='topic-tag'>"
                        f"<span class='{css}'>{icon}</span> {label}{assertion}"
                        f"</span>"
                    )
                st.markdown(
                    "<div style='margin:8px 0 4px 0;font-size:12px;font-weight:600;"
                    "color:#475569;'>Topics detected in your review:</div>"
                    "<div style='display:flex;flex-wrap:wrap;gap:4px;margin-bottom:12px;'>"
                    + "".join(chips)
                    + "</div>",
                    unsafe_allow_html=True,
                )

    # Follow-up questions
    if st.session_state.pending_questions:
        st.markdown("<div class='section-hdr'>Follow-up questions</div>", unsafe_allow_html=True)
        _fc = build_field_cluster_map(topics)
        display_slot = 0
        for idx, (q, answered) in enumerate(st.session_state.pending_questions):
            if answered:
                continue

            st.markdown(
                f"<div class='q-card'>"
                f"<div class='q-text'>{q.question_text}</div>"
                f"<div class='q-reason'>Why we asked: {q.reason}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # "Why this question?" expander with rich explanation
            sf_match = sf_lookup.get(q.field_id) if result else None
            if sf_match:
                with st.expander("Why this question?", expanded=False):
                    cluster_label = CLUSTER_LABELS.get(sf_match.cluster or "", sf_match.cluster or "").title()
                    st.markdown(
                        f"<span class='rank-badge'>Rank #{sf_match.rank} of {result.total_fields}</span>"
                        f"<span class='cluster-badge'>{cluster_label}</span>",
                        unsafe_allow_html=True,
                    )

                    # Plain-English explanation
                    explanation = _rich_why(
                        sf_match,
                        display_slot=display_slot,
                        all_questions=st.session_state.pending_questions,
                        sf_lookup=sf_lookup,
                        field_cluster=_fc,
                    )
                    st.markdown(
                        f"<div style='font-size:13px;color:#475569;line-height:1.6;padding:6px 0;'>"
                        f"{explanation}</div>",
                        unsafe_allow_html=True,
                    )

                    # Component score bars
                    st.markdown(
                        f"<div style='margin:8px 0;'>{_component_bars_html(sf_match)}</div>",
                        unsafe_allow_html=True,
                    )

                    # Review overlap (which topics were de-prioritized)
                    overlap = _review_overlap_html(result, topics)
                    if overlap:
                        st.markdown(overlap, unsafe_allow_html=True)

                    # Runner-ups
                    runners = _runner_ups_html(sf_match, result)
                    if runners:
                        st.markdown(runners, unsafe_allow_html=True)

                    # Impact preview
                    impact = _impact_preview_html(sf_match, sf_lookup)
                    if impact:
                        st.markdown(impact, unsafe_allow_html=True)

                    # Raw formula
                    st.markdown(
                        f"<div class='score-detail'>"
                        f"Score = <span class='val'>{sf_match.composite:.3f}</span><br>"
                        f"&nbsp;&nbsp;= 0.55 \u00d7 no_data(<span class='val'>{sf_match.missing:.2f}</span>)"
                        f" + 0.25 \u00d7 outdated(<span class='val'>{sf_match.stale:.2f}</span>)"
                        f" + 0.15 \u00d7 rarely_discussed(<span class='val'>{sf_match.coverage:.2f}</span>)"
                        f" \u2212 0.35 \u00d7 redundancy(<span class='val'>{sf_match.redundancy:.2f}</span>)"
                        f" + 0.02"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            display_slot += 1

            # Input widget
            input_key = f"answer_input_{idx}"
            if q.input_type == "rating_1_5":
                answer_value = st.select_slider(
                    "Rate 1\u20135", options=[1, 2, 3, 4, 5], value=3, key=input_key,
                )
                submit_text = str(answer_value)
            elif q.input_type == "yes_no":
                answer_value = st.radio(
                    "Answer", ["Yes", "No"], key=input_key, horizontal=True,
                )
                submit_text = answer_value.lower()
            else:
                submit_text = ""
                if MIC_AVAILABLE:
                    voice = speech_to_text(
                        language="en",
                        start_prompt="\U0001f3a4 Answer by voice",
                        stop_prompt="\u23f9 Stop",
                        just_once=True,
                        use_container_width=False,
                        key=f"voice_{idx}",
                    )
                    if voice:
                        st.session_state[f"text_{idx}"] = voice
                        st.session_state[input_key] = voice
                submit_text = st.text_input(
                    "Your answer",
                    value=st.session_state.get(f"text_{idx}", ""),
                    key=input_key,
                )

            bcol1, bcol2, _ = st.columns([1, 1, 3])
            with bcol1:
                if st.button("Submit answer", key=f"submit_{idx}"):
                    active_pid = st.session_state.get("active_property_id") or property_id
                    if active_pid not in flow._pending_review_id:
                        saved_id = st.session_state.get("pending_review_id")
                        if saved_id:
                            flow._pending_review_id[active_pid] = saved_id
                    with st.spinner("Updating property info..."):
                        answer = flow.submit_answer(active_pid, q, submit_text, today=today)
                    st.session_state.pending_questions[idx] = (q, True)
                    if answer.status == "scored":
                        st.session_state.last_flashed_field = q.field_id
                        st.success(f"Saved: {answer.parsed_value}")
                    elif answer.status == "unscorable":
                        st.warning("Couldn't parse that answer \u2014 recording as unscorable.")
                    st.rerun()
            with bcol2:
                if st.button("Skip", key=f"skip_{idx}"):
                    active_pid = st.session_state.get("active_property_id") or property_id
                    flow.submit_answer(active_pid, q, None, today=today)
                    st.session_state.pending_questions[idx] = (q, True)
                    st.rerun()

        # Completion message
        if all(a for _, a in st.session_state.pending_questions):
            st.session_state.active_property_id = None
            st.success("All follow-ups addressed. Submit another review to continue.")
