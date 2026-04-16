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
from src.ranker import build_field_cluster_map
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
    """Return HTML for a tiny stacked score bar."""
    total = sf.missing + sf.stale + sf.coverage + 0.001
    m_pct = sf.missing / total * 100
    s_pct = sf.stale / total * 100
    c_pct = sf.coverage / total * 100
    return (
        f"<span class='score-bar-wrap'>"
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
    if st.button("Reset", help="Clear session answers (DB untouched)", use_container_width=True):
        for k in [
            "pending_questions", "answered_fields", "last_flashed_field",
            "review_text", "active_property_id", "submit_result",
        ]:
            st.session_state.pop(k, None)
        _init_state()
        st.rerun()

prop = repo.get_property(property_id)
today = _max_review_date(repo)

# ---------- ScoredField lookup from stored SubmitResult ----------
result: SubmitResult | None = st.session_state.get("submit_result")
sf_lookup: dict[str, ScoredField] = {}
if result:
    for sf in result.scored_fields:
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
        st.plotly_chart(make_donut_chart(known, total), use_container_width=True, key="donut")
        st.markdown(
            f"<div class='coverage-label'>{known} / {total} fields known</div>",
            unsafe_allow_html=True,
        )
    with chart_right:
        st.plotly_chart(
            make_radar_chart(all_fs, topics, property_id),
            use_container_width=True,
            key="radar",
        )

    # Sub-ratings section
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
    else:
        st.caption("(Install `streamlit-mic-recorder` for voice input.)")

    review_text = st.text_area(
        "Your review (or leave blank for cold-start demo)",
        value=st.session_state.review_text,
        key="review_input",
        height=120,
        label_visibility="collapsed",
        placeholder="Write your review here, or leave blank to see cold-start gap analysis...",
    )

    if st.button("Submit review", type="primary", key="submit_review", use_container_width=True):
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

    # Follow-up questions
    if st.session_state.pending_questions:
        st.markdown("<div class='section-hdr'>Follow-up questions</div>", unsafe_allow_html=True)
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

            # "Why this question?" expander with score breakdown
            sf_match = sf_lookup.get(q.field_id) if result else None
            if sf_match:
                with st.expander("Why this question?", expanded=False):
                    st.markdown(
                        f"<span class='rank-badge'>Rank #{sf_match.rank} of {result.total_fields}</span>"
                        f"<span class='cluster-badge'>{(sf_match.cluster or 'Unknown').title()}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='score-detail'>"
                        f"Score = <span class='val'>{sf_match.composite:.3f}</span><br>"
                        f"&nbsp;&nbsp;= 0.55 \u00d7 missing(<span class='val'>{sf_match.missing:.2f}</span>)"
                        f" + 0.25 \u00d7 stale(<span class='val'>{sf_match.stale:.2f}</span>)"
                        f" + 0.15 \u00d7 coverage(<span class='val'>{sf_match.coverage:.2f}</span>)"
                        f" \u2212 0.35 \u00d7 redundancy(<span class='val'>{sf_match.redundancy:.2f}</span>)"
                        f" + 0.02"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

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
