from __future__ import annotations
import sys
from pathlib import Path

# Make 'src' importable regardless of where streamlit is launched from.
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
from datetime import date
import numpy as np
from src.db import Repo
from src.llm import LlmClient
from src.taxonomy import load_taxonomy, SCHEMA_DESCRIPTION_FIELDS
from src.ranker import build_field_cluster_map
from src.flow import AskFlow
from src.models import SUB_RATING_KEYS

try:
    from streamlit_mic_recorder import speech_to_text
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

# ---------- page config ----------
st.set_page_config(page_title="Ask What Matters", layout="wide", page_icon="💬")

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
.field-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #eee; }
.field-known { color: #16a34a; font-weight: 600; }
.field-unknown { color: #9ca3af; }
.flash { animation: flash 1.8s ease-out; background: transparent; }
@keyframes flash { 0% { background: #fef08a; } 100% { background: transparent; } }
.reason { color: #6b7280; font-size: 13px; font-style: italic; }
.coverage-big { font-size: 28px; font-weight: 700; }
.question-card { padding: 16px; border: 2px solid #6366f1; border-radius: 12px; margin: 12px 0; background: #fafaff; }
</style>
""", unsafe_allow_html=True)

# ---------- state init ----------
def _init_state():
    ss = st.session_state
    ss.setdefault("property_id", None)
    ss.setdefault("review_text", "")
    ss.setdefault("pending_questions", [])
    ss.setdefault("answered_fields", set())  # for the flash animation
    ss.setdefault("last_flashed_field", None)
    ss.setdefault("answer_widgets", {})  # per-question widget key tracking
_init_state()

# ---------- property picker ----------
repo = get_repo()
flow = get_flow()
topics = get_taxonomy()

properties = repo.list_properties()
property_options = {
    f"{p.city or '?'}, {p.country or '?'} "
    f"({p.star_rating or '?'}★, {len(repo.list_reviews_for(p.eg_property_id))} reviews)": p.eg_property_id
    for p in properties
}

header_cols = st.columns([4, 1, 1])
with header_cols[0]:
    st.title("Ask What Matters")
    st.caption("Adaptive follow-up questions that fill missing property information.")
with header_cols[1]:
    picked = st.selectbox("Property", list(property_options.keys()), key="property_picker")
    st.session_state.property_id = property_options[picked]
with header_cols[2]:
    if st.button("🔄 Reset", help="Clear this session's answers (DB untouched)"):
        for k in ["pending_questions", "answered_fields", "last_flashed_field", "review_text"]:
            st.session_state.pop(k, None)
        _init_state()
        st.rerun()

property_id = st.session_state.property_id
prop = repo.get_property(property_id)

# ---------- info card (left) + review area (right) ----------
left, right = st.columns([2, 3])

with left:
    st.subheader("Property info")
    st.markdown(f"**{prop.city}, {prop.country}** · {prop.star_rating or '?'}★ · "
                f"guest avg {prop.guestrating_avg_expedia or '?'}")
    if prop.check_in_start_time:
        st.markdown(f"Check-in: {prop.check_in_start_time} – {prop.check_in_end_time} · "
                    f"Check-out: {prop.check_out_time or '?'}")

    # Sub-ratings
    st.markdown("##### Sub-ratings")
    flash_field = st.session_state.last_flashed_field
    for key in SUB_RATING_KEYS:
        fid = f"rating:{key}"
        fs = repo.get_field_state(property_id, fid)
        pretty = key.replace("_", " ").title()
        css_class = "flash" if flash_field == fid else ""
        if fs and fs.value_known:
            ema = f"{fs.short_ema:.1f}" if fs.short_ema is not None else "?"
            st.markdown(
                f"<div class='field-row {css_class}'>"
                f"<span>{pretty}</span>"
                f"<span class='field-known'>{ema} / 5</span>"
                f"</div>", unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='field-row {css_class}'>"
                f"<span>{pretty}</span>"
                f"<span class='field-unknown'>— not rated</span>"
                f"</div>", unsafe_allow_html=True,
            )

    # Schema coverage (amenities + policies)
    st.markdown("##### Description fields")
    for sf in SCHEMA_DESCRIPTION_FIELDS:
        fid = f"schema:{sf}"
        fs = repo.get_field_state(property_id, fid)
        pretty = sf.replace("property_amenity_", "").replace("_", " ").title()
        css_class = "flash" if flash_field == fid else ""
        if fs and fs.value_known:
            st.markdown(
                f"<div class='field-row {css_class}'>"
                f"<span>{pretty}</span>"
                f"<span class='field-known'>✓ filled</span>"
                f"</div>", unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='field-row {css_class}'>"
                f"<span>{pretty}</span>"
                f"<span class='field-unknown'>— empty</span>"
                f"</div>", unsafe_allow_html=True,
            )

with right:
    # Coverage meter
    all_fs = repo.list_field_states_for(property_id)
    known = sum(1 for fs in all_fs if fs.value_known)
    total = len(all_fs)
    st.markdown(
        f"<div style='text-align: right;'>"
        f"<div>Coverage</div>"
        f"<div class='coverage-big'>{known} / {total}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Write your review")

    # Voice input
    if MIC_AVAILABLE:
        voice_text = speech_to_text(
            language="en", start_prompt="🎤 Record review", stop_prompt="⏹ Stop",
            just_once=True, use_container_width=False, key="voice_review",
        )
        if voice_text:
            st.session_state.review_text = voice_text
    else:
        st.caption("(Install `streamlit-mic-recorder` for voice input.)")

    review_text = st.text_area(
        "Your review (or leave blank for cold-start demo)",
        value=st.session_state.review_text,
        key="review_input", height=100,
    )

    if st.button("Submit review", type="primary", key="submit_review"):
        today = _max_review_date(repo)
        with st.spinner("Analyzing your review and picking the best follow-up..."):
            questions = flow.submit_review(property_id, review_text, today=today)
        st.session_state.pending_questions = [(q, False) for q in questions]
        st.session_state.review_text = review_text
        # Persist pending review_id in session_state (survives cache eviction)
        st.session_state.pending_review_id = flow._pending_review_id.get(property_id)
        st.rerun()

    # Follow-up questions
    if st.session_state.pending_questions:
        st.markdown("### Follow-up questions")
        for idx, (q, answered) in enumerate(st.session_state.pending_questions):
            if answered:
                continue

            with st.container():
                st.markdown(
                    f"<div class='question-card'>"
                    f"<div style='font-size: 18px; font-weight: 600;'>{q.question_text}</div>"
                    f"<div class='reason'>Why we asked: {q.reason}</div>"
                    f"</div>", unsafe_allow_html=True,
                )

                input_key = f"answer_input_{idx}"
                if q.input_type == "rating_1_5":
                    answer_value = st.select_slider(
                        "Rate 1–5", options=[1, 2, 3, 4, 5], value=3, key=input_key,
                    )
                    submit_text = str(answer_value)
                elif q.input_type == "yes_no":
                    answer_value = st.radio("Answer", ["Yes", "No"], key=input_key, horizontal=True)
                    submit_text = answer_value.lower()
                else:
                    submit_text = ""
                    if MIC_AVAILABLE:
                        voice = speech_to_text(
                            language="en", start_prompt="🎤 Answer by voice",
                            stop_prompt="⏹ Stop", just_once=True,
                            use_container_width=False, key=f"voice_{idx}",
                        )
                        if voice:
                            st.session_state[f"text_{idx}"] = voice
                    submit_text = st.text_input(
                        "Your answer", value=st.session_state.get(f"text_{idx}", ""),
                        key=input_key,
                    )

                bcol1, bcol2, _ = st.columns([1, 1, 3])
                with bcol1:
                    if st.button("Submit answer", key=f"submit_{idx}"):
                        # Restore pending review_id from session_state if cache was evicted
                        if property_id not in flow._pending_review_id:
                            saved_id = st.session_state.get("pending_review_id")
                            if saved_id:
                                flow._pending_review_id[property_id] = saved_id
                        today = _max_review_date(repo)
                        with st.spinner("Updating property info..."):
                            answer = flow.submit_answer(property_id, q, submit_text, today=today)
                        st.session_state.pending_questions[idx] = (q, True)
                        if answer.status == "scored":
                            st.session_state.last_flashed_field = q.field_id
                            st.success(f"Saved: {answer.parsed_value}")
                        elif answer.status == "unscorable":
                            st.warning("Couldn't parse that answer — recording as unscorable.")
                        st.rerun()
                with bcol2:
                    if st.button("Skip", key=f"skip_{idx}"):
                        today = _max_review_date(repo)
                        flow.submit_answer(property_id, q, None, today=today)
                        st.session_state.pending_questions[idx] = (q, True)
                        st.rerun()

        # If all answered, show a summary
        if all(a for _, a in st.session_state.pending_questions):
            st.success("✅ All follow-ups addressed. Submit another review to continue.")
