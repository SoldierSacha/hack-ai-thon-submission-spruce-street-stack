# UI Redesign: Coverage Radar Dashboard — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign the Streamlit UI into an integrated dashboard that surfaces the question engine's gap analysis, scoring, and enrichment pipeline for hackathon judges.

**Architecture:** Add `ScoredField` and `SubmitResult` models so `flow.submit_review()` returns enrichment metadata + per-field score breakdowns alongside questions. Rewrite `streamlit_app.py` with Plotly radar/donut charts, color-coded field inventory, score breakdowns on question cards, and Expedia-branded styling.

**Tech Stack:** Streamlit, Plotly (new dep), existing Python/Pydantic/SQLite stack.

---

### Task 1: Add `ScoredField` model

**Files:**
- Modify: `question-engine/src/models.py:85` (insert before `Question`)
- Test: `question-engine/tests/test_models.py`

**Step 1: Write the failing test**

Add to `question-engine/tests/test_models.py`:

```python
from src.models import ScoredField, FieldState

def test_scored_field_stores_component_scores():
    fs = FieldState(eg_property_id="p1", field_id="topic:wifi", value_known=False)
    sf = ScoredField(
        field_state=fs, composite=0.72,
        missing=1.0, stale=0.0, coverage=0.85, redundancy=0.12,
        rank=1, cluster="connectivity",
    )
    assert sf.composite == 0.72
    assert sf.missing == 1.0
    assert sf.rank == 1
    assert sf.cluster == "connectivity"
    assert sf.field_state.field_id == "topic:wifi"
```

**Step 2: Run test to verify it fails**

Run: `cd question-engine && python -m pytest tests/test_models.py::test_scored_field_stores_component_scores -v`
Expected: FAIL — `ImportError: cannot import name 'ScoredField'`

**Step 3: Write minimal implementation**

Insert in `question-engine/src/models.py` at line 85 (before `class Question`):

```python
class ScoredField(BaseModel):
    field_state: FieldState
    composite: float
    missing: float
    stale: float
    coverage: float
    redundancy: float
    rank: int = 0
    cluster: str = ""
```

**Step 4: Run test to verify it passes**

Run: `cd question-engine && python -m pytest tests/test_models.py::test_scored_field_stores_component_scores -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: add ScoredField model for score breakdown visibility"
```

---

### Task 2: Add `EnrichmentMeta` and `SubmitResult` models

**Files:**
- Modify: `question-engine/src/models.py` (append after `ScoredField`)
- Test: `question-engine/tests/test_models.py`

**Step 1: Write the failing test**

Add to `question-engine/tests/test_models.py`:

```python
from src.models import EnrichmentMeta, SubmitResult, Question, ScoredField, FieldState

def test_submit_result_bundles_questions_and_metadata():
    meta = EnrichmentMeta(lang="de", translated=True, embedding_dim=384, topics_tagged=8, topics_total=28)
    q = Question(field_id="rating:checkin", question_text="How was check-in?",
                 input_type="rating_1_5", reason="No data")
    fs = FieldState(eg_property_id="p1", field_id="rating:checkin", value_known=False)
    sf = ScoredField(field_state=fs, composite=0.72, missing=1.0, stale=0.0,
                     coverage=0.85, redundancy=0.0, rank=1, cluster="service")
    result = SubmitResult(questions=[q], scored_fields=[sf], enrichment=meta, total_fields=58)
    assert len(result.questions) == 1
    assert result.enrichment.lang == "de"
    assert result.enrichment.translated is True
    assert result.total_fields == 58
```

**Step 2: Run test to verify it fails**

Run: `cd question-engine && python -m pytest tests/test_models.py::test_submit_result_bundles_questions_and_metadata -v`
Expected: FAIL — `ImportError: cannot import name 'EnrichmentMeta'`

**Step 3: Write minimal implementation**

Append to `question-engine/src/models.py` after `ScoredField`:

```python
class EnrichmentMeta(BaseModel):
    lang: str = "unknown"
    translated: bool = False
    embedding_dim: int = 0
    topics_tagged: int = 0
    topics_total: int = 0

class SubmitResult(BaseModel):
    questions: list[Question]
    scored_fields: list[ScoredField]
    enrichment: EnrichmentMeta
    total_fields: int = 0
```

**Step 4: Run test to verify it passes**

Run: `cd question-engine && python -m pytest tests/test_models.py::test_submit_result_bundles_questions_and_metadata -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: add EnrichmentMeta and SubmitResult models"
```

---

### Task 3: Modify `rank_fields()` to return `ScoredField` objects

**Files:**
- Modify: `question-engine/src/ranker.py:16-48` (the `rank_fields` function)
- Modify: `question-engine/src/ranker.py:51-68` (the `pick_k` function — update type hints)
- Test: `question-engine/tests/test_ranker.py`

**Step 1: Write the failing test**

Add to `question-engine/tests/test_ranker.py`:

```python
from src.models import ScoredField

def test_rank_fields_returns_scored_field_objects():
    states = [
        _fs(field_id="rating:checkin", value_known=False),
        _fs(field_id="rating:overall", value_known=True, mention_count=100),
    ]
    ranked = rank_fields(
        property_id="p1", field_states=states,
        today=date(2025, 9, 1), topic_embeddings={}, review_embedding=None,
        field_cluster={"rating:checkin": "service", "rating:overall": "overall"},
        weights_path=WEIGHTS,
    )
    assert len(ranked) == 2
    sf = ranked[0]
    assert isinstance(sf, ScoredField)
    assert sf.field_state.field_id == "rating:checkin"
    assert sf.composite > 0
    assert sf.missing == 1.0  # value_known=False
    assert sf.rank == 1
    assert sf.cluster == "service"
```

**Step 2: Run test to verify it fails**

Run: `cd question-engine && python -m pytest tests/test_ranker.py::test_rank_fields_returns_scored_field_objects -v`
Expected: FAIL — `ranked[0]` is a tuple, not a `ScoredField`

**Step 3: Modify `rank_fields` and `pick_k`**

Replace the body of `rank_fields()` in `question-engine/src/ranker.py:16-48` with:

```python
def rank_fields(
    *,
    property_id: str,
    field_states: list[FieldState],
    today: date,
    topic_embeddings: dict[str, np.ndarray],
    review_embedding: np.ndarray | None,
    field_cluster: dict[str, str],
    weights_path: str = "config/weights.yaml",
) -> list[ScoredField]:
    """Score each field_state for `property_id` and return ScoredField objects sorted by descending score."""
    W = yaml.safe_load(Path(weights_path).read_text())
    scored: list[ScoredField] = []
    for fs in field_states:
        if fs.eg_property_id != property_id:
            continue
        m = missing_score(fs)
        s = stale_score(fs, today=today, time_horizon_days=W["time_horizon_days"])
        c = coverage_gap_score(fs)
        red = 0.0
        if review_embedding is not None and fs.field_id.startswith("topic:"):
            topic_id = fs.field_id.split(":", 1)[1]
            if topic_id in topic_embeddings:
                red = max(0.0, _cosine(review_embedding, topic_embeddings[topic_id]))
        composite = (
            W["w_missing"] * m
            + W["w_stale"] * s
            + W["w_coverage"] * c
            - W["w_redundancy"] * red
            + W["epsilon"]
        )
        scored.append(ScoredField(
            field_state=fs, composite=composite,
            missing=m, stale=s, coverage=c, redundancy=red,
            cluster=field_cluster.get(fs.field_id, ""),
        ))
    scored.sort(key=lambda x: -x.composite)
    for i, sf in enumerate(scored):
        sf.rank = i + 1
    return scored
```

Update the import at the top of `ranker.py` (line 8):

```python
from src.models import FieldState, TaxonomyTopic, ScoredField
```

Replace `pick_k` in `question-engine/src/ranker.py:51-68` with:

```python
def pick_k(
    ranked: list[ScoredField],
    field_cluster: dict[str, str],
    min_score_for_k2: float = 0.40,
) -> list[ScoredField]:
    """Pick at most 2 fields from a ranked list, skipping same-cluster duplicates for slot 2."""
    if not ranked:
        return []
    picks = [ranked[0]]
    first_cluster = ranked[0].cluster or field_cluster.get(ranked[0].field_state.field_id)
    for sf in ranked[1:]:
        if sf.composite < min_score_for_k2:
            break
        sf_cluster = sf.cluster or field_cluster.get(sf.field_state.field_id)
        if sf_cluster == first_cluster:
            continue
        picks.append(sf)
        break
    return picks
```

**Step 4: Update existing ranker tests**

The existing tests reference `ranked[0][0]` (tuple indexing) and `pick_k` with `(fs, score)` tuples. Update:

In `question-engine/tests/test_ranker.py`, update all existing tests:

- `ranked[0][0]` → `ranked[0].field_state`
- `ranked[0][1]` → `ranked[0].composite`
- `ranked[1][1]` → `ranked[1].composite`
- `pick_k` call with tuple lists → use `ScoredField` objects:

Replace `test_pick_k_returns_one_when_second_score_too_low`:

```python
def test_pick_k_returns_one_when_second_score_too_low():
    ranked = [
        ScoredField(field_state=_fs(field_id="rating:checkin"), composite=0.6,
                     missing=1.0, stale=0.0, coverage=0.0, redundancy=0.0, cluster="service"),
        ScoredField(field_state=_fs(field_id="rating:service"), composite=0.2,
                     missing=0.0, stale=0.0, coverage=0.0, redundancy=0.0, cluster="service"),
    ]
    picks = pick_k(ranked, field_cluster={"rating:checkin": "service", "rating:service": "service"})
    assert len(picks) == 1
    assert picks[0].field_state.field_id == "rating:checkin"
```

Replace `test_pick_k_dedupes_same_cluster`:

```python
def test_pick_k_dedupes_same_cluster():
    ranked = [
        ScoredField(field_state=_fs(field_id="rating:roomcleanliness"), composite=0.8,
                     missing=1.0, stale=0.0, coverage=0.0, redundancy=0.0, cluster="room"),
        ScoredField(field_state=_fs(field_id="rating:roomcomfort"), composite=0.7,
                     missing=1.0, stale=0.0, coverage=0.0, redundancy=0.0, cluster="room"),
        ScoredField(field_state=_fs(field_id="rating:service"), composite=0.6,
                     missing=1.0, stale=0.0, coverage=0.0, redundancy=0.0, cluster="service"),
    ]
    cluster = {
        "rating:roomcleanliness": "room",
        "rating:roomcomfort": "room",
        "rating:service": "service",
    }
    picks = pick_k(ranked, field_cluster=cluster, min_score_for_k2=0.4)
    assert len(picks) == 2
    ids = [p.field_state.field_id for p in picks]
    assert ids == ["rating:roomcleanliness", "rating:service"]
```

**Step 5: Run full ranker test suite**

Run: `cd question-engine && python -m pytest tests/test_ranker.py -v`
Expected: all PASS

**Step 6: Commit**

```bash
git add src/ranker.py tests/test_ranker.py
git commit -m "refactor: rank_fields returns ScoredField with component score breakdown"
```

---

### Task 4: Modify `flow.submit_review()` to return `SubmitResult`

**Files:**
- Modify: `question-engine/src/flow.py:9-11` (imports), `question-engine/src/flow.py:43-128` (`submit_review` method)
- Test: `question-engine/tests/test_flow.py`

**Step 1: Write the failing test**

Add to `question-engine/tests/test_flow.py`:

```python
from src.models import SubmitResult

def test_flow_submit_review_returns_submit_result(tmp_path):
    flow, repo, llm = _setup_flow(tmp_path)
    result = flow.submit_review("p1", "The wifi was excellent and fast", date(2025, 9, 1))
    assert isinstance(result, SubmitResult)
    assert 1 <= len(result.questions) <= 2
    assert result.enrichment.lang != ""
    assert result.total_fields > 0
    assert len(result.scored_fields) == result.total_fields
    # Each scored field has component scores
    for sf in result.scored_fields:
        assert sf.rank >= 1
        assert sf.composite >= 0 or sf.composite < 0  # can be negative from redundancy
```

**Step 2: Run test to verify it fails**

Run: `cd question-engine && python -m pytest tests/test_flow.py::test_flow_submit_review_returns_submit_result -v`
Expected: FAIL — `submit_review` returns a list, not `SubmitResult`

**Step 3: Modify `flow.submit_review()`**

Update imports at top of `question-engine/src/flow.py` (line 10):

```python
from src.models import (
    Answer, FieldState, Property, Question, Review, RatingBreakdown,
    TaxonomyTopic, ScoredField, EnrichmentMeta, SubmitResult,
)
```

Replace `submit_review` method body (lines 43-128). The method signature changes:

```python
    def submit_review(
        self, property_id: str, review_text: str, today: date
    ) -> SubmitResult:
        # 1. Build a synthetic review_id
        review_id = f"{property_id}:live:{int(time.time() * 1000)}"

        # 2. Process the review text
        lang = detect_language(review_text) if review_text else "unknown"
        text_en = (
            translate_to_english(review_text, lang, self.llm) if review_text else None
        )

        # 3. Compute embedding (only if we have text)
        review_embedding = None
        embedding_dim = 0
        if text_en and text_en.strip():
            vec = self.llm.embed(text_en)
            review_embedding = np.array(vec, dtype=np.float32)
            embedding_dim = len(vec)

        # 4. Tag against taxonomy
        if text_en:
            tags = tag_review(text_en, self.taxonomy, self.llm)
        else:
            tags = {
                t.topic_id: {"mentioned": False, "sentiment": None, "assertion": None}
                for t in self.taxonomy
            }

        # 5. Persist the review + embedding + tags
        rev = Review(
            review_id=review_id,
            eg_property_id=property_id,
            acquisition_date=today,
            rating=RatingBreakdown(),
            review_text_orig=review_text,
            review_text_en=text_en,
            lang=lang,
            source="live",
        )
        self.repo.upsert_review(rev)
        if review_embedding is not None:
            self.repo.set_embedding(review_id, review_embedding)
        tag_list = [
            {
                "field_id": f"topic:{tid}",
                "mentioned": bool(v.get("mentioned")),
                "sentiment": v.get("sentiment"),
                "assertion": v.get("assertion"),
            }
            for tid, v in tags.items()
        ]
        self.repo.upsert_review_tags(review_id, tag_list)

        # 6. Re-aggregate field_state for this property
        self._reaggregate_property(property_id)

        # 7. Rank and pick
        states = self.repo.list_field_states_for(property_id)
        ranked = rank_fields(
            property_id=property_id,
            field_states=states,
            today=today,
            topic_embeddings=self.topic_embeddings,
            review_embedding=review_embedding,
            field_cluster=self.field_cluster,
            weights_path=self.weights_path,
        )
        picks = pick_k(ranked, field_cluster=self.field_cluster)

        # 8. Render each picked field into a Question
        prop = self.repo.get_property(property_id)
        questions: list[Question] = []
        for sf in picks:
            fs = sf.field_state
            topic = None
            if fs.field_id.startswith("topic:"):
                topic = self.topic_by_id.get(fs.field_id.split(":", 1)[1])
            q = render_question(
                field_state=fs, property_=prop, topic=topic, llm=self.llm
            )
            questions.append(q)

        # Remember which review this set of Questions belongs to.
        self._pending_review_id[property_id] = review_id

        # 9. Build enrichment metadata
        topics_tagged = sum(1 for v in tags.values() if v.get("mentioned"))
        enrichment = EnrichmentMeta(
            lang=lang,
            translated=(lang not in ("en", "unknown") and text_en is not None),
            embedding_dim=embedding_dim,
            topics_tagged=topics_tagged,
            topics_total=len(self.taxonomy),
        )

        return SubmitResult(
            questions=questions,
            scored_fields=ranked,
            enrichment=enrichment,
            total_fields=len(ranked),
        )
```

**Step 4: Update existing flow tests**

The existing tests call `flow.submit_review()` and expect a list. Update them:

In `test_flow_submit_review_returns_questions`:
```python
def test_flow_submit_review_returns_questions(tmp_path):
    flow, repo, llm = _setup_flow(tmp_path)
    today = date(2025, 9, 1)
    result = flow.submit_review("p1", "", today)
    assert 1 <= len(result.questions) <= 2
    for q in result.questions:
        assert q.field_id
        assert q.question_text
```

In `test_flow_empty_review_skips_translate_and_embed`:
```python
def test_flow_empty_review_skips_translate_and_embed(tmp_path):
    flow, repo, llm = _setup_flow(tmp_path)
    flow.submit_review("p1", "", date(2025, 9, 1))
    llm.embed.assert_not_called()
```
(No change needed — return value isn't used.)

In `test_flow_submit_review_persists_with_live_source`:
```python
def test_flow_submit_review_persists_with_live_source(tmp_path):
    flow, repo, llm = _setup_flow(tmp_path)
    flow.submit_review("p1", "Great stay", date(2025, 9, 1))
    reviews = repo.list_reviews_for("p1")
    assert len(reviews) == 1
    assert reviews[0].source == "live"
```
(No change needed — return value isn't used.)

In `test_flow_submit_answer_scored_updates_field_state`:
```python
def test_flow_submit_answer_scored_updates_field_state(tmp_path):
    flow, repo, llm = _setup_flow(tmp_path)
    result = flow.submit_review("p1", "", date(2025, 9, 1))
    q = result.questions[0]
    # ... rest unchanged
```

In `test_flow_submit_answer_skipped_does_not_update_state`:
```python
def test_flow_submit_answer_skipped_does_not_update_state(tmp_path):
    flow, repo, llm = _setup_flow(tmp_path)
    result = flow.submit_review("p1", "", date(2025, 9, 1))
    q = result.questions[0]
    # ... rest unchanged
```

**Step 5: Run full flow + ranker test suites**

Run: `cd question-engine && python -m pytest tests/test_flow.py tests/test_ranker.py -v`
Expected: all PASS

**Step 6: Commit**

```bash
git add src/flow.py tests/test_flow.py
git commit -m "feat: submit_review returns SubmitResult with enrichment metadata and score breakdowns"
```

---

### Task 5: Add plotly dependency

**Files:**
- Modify: `question-engine/pyproject.toml:6-16` (dependencies list)

**Step 1: Add plotly**

In `question-engine/pyproject.toml`, add `"plotly>=5.18"` to the `dependencies` list.

**Step 2: Install**

Run: `cd question-engine && pip install -e .`
Expected: plotly installs successfully

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add plotly for radar and donut charts"
```

---

### Task 6: Rewrite `streamlit_app.py` — CSS and helpers

This is the start of the UI rewrite. Create the foundation: global CSS, helper functions, and cached resources.

**Files:**
- Modify: `question-engine/app/streamlit_app.py` (full rewrite)

**Step 1: Replace the CSS block and add all helper functions**

Replace the entire file. The new structure is:

```python
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
st.set_page_config(page_title="Ask What Matters", layout="wide", page_icon="🔍")

# ---------- resources ----------
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
    all_dates = []
    for p in repo.list_properties():
        for r in repo.list_reviews_for(p.eg_property_id):
            all_dates.append(r.acquisition_date)
    return max(all_dates) if all_dates else date.today()

# ---------- CSS ----------
st.markdown("""
<style>
/* Header */
.header-bar {
    background: linear-gradient(135deg, #00355F 0%, #004E8C 100%);
    padding: 20px 28px;
    border-radius: 0 0 16px 16px;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex; align-items: center; justify-content: space-between;
}
.header-bar h1 { color: white; margin: 0; font-size: 26px; font-weight: 700; }
.header-bar p { color: rgba(255,255,255,0.8); margin: 4px 0 0 0; font-size: 14px; }

/* Cards */
.prop-card {
    background: white; border-radius: 12px; padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 16px;
}
.prop-card .city { font-size: 20px; font-weight: 700; color: #1e293b; }
.prop-card .meta { font-size: 13px; color: #64748b; margin-top: 4px; }

/* Stars */
.star-pill {
    display: inline-block; background: #FBCE38; color: #00355F;
    padding: 2px 10px; border-radius: 999px; font-size: 13px; font-weight: 700;
}

/* Field rows */
.field-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid #f1f5f9; font-size: 14px;
}
.field-row:last-child { border-bottom: none; }

/* Status pills */
.pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 11px; font-weight: 600; }
.pill-known { background: #dcfce7; color: #16a34a; }
.pill-stale { background: #fef3c7; color: #d97706; }
.pill-drifting { background: #fef3c7; color: #d97706; }
.pill-missing { background: #fee2e2; color: #ef4444; }

/* Flash animation */
.flash { animation: flash 1.8s ease-out; }
@keyframes flash { 0% { background: #fef08a; } 100% { background: transparent; } }

/* Score bar */
.score-bar-wrap { width: 80px; display: inline-flex; align-items: center; gap: 4px; }
.score-bar { display: flex; height: 5px; border-radius: 3px; overflow: hidden; flex: 1; background: #e5e7eb; }
.seg-m { background: #ef4444; }
.seg-s { background: #f59e0b; }
.seg-c { background: #6366f1; }

/* Question card */
.q-card {
    border: 2px solid #6366f1; border-radius: 12px; padding: 20px;
    margin: 12px 0; background: #fafaff;
}
.q-card .q-text { font-size: 18px; font-weight: 600; color: #1e293b; margin-bottom: 8px; }
.q-card .q-reason { color: #6b7280; font-size: 13px; font-style: italic; }

/* Score detail inside question card */
.score-detail { font-family: monospace; font-size: 12px; color: #475569; margin: 6px 0; }
.score-detail .val { font-weight: 700; }
.rank-badge {
    display: inline-block; background: #6366f1; color: white;
    padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
}
.cluster-badge {
    display: inline-block; background: #f0f0ff; color: #6366f1;
    padding: 2px 10px; border-radius: 999px; font-size: 12px; margin-left: 6px;
}

/* Enrichment strip */
.enrich-strip { display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0; }
.enrich-chip {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px;
    padding: 6px 14px; font-size: 13px; color: #166534;
}
.enrich-chip.translated { background: #eff6ff; border-color: #bfdbfe; color: #1e40af; }
.enrich-chip.embedding { background: #faf5ff; border-color: #e9d5ff; color: #6b21a8; }
.enrich-chip.topics { background: #fefce8; border-color: #fde68a; color: #854d0e; }

/* Coverage big number */
.coverage-big { font-size: 32px; font-weight: 800; color: #00355F; text-align: center; }
.coverage-label { font-size: 13px; color: #64748b; text-align: center; }

/* Section headers */
.section-hdr { font-size: 14px; font-weight: 700; color: #00355F; text-transform: uppercase; letter-spacing: 0.5px; margin: 16px 0 8px 0; }
</style>
""", unsafe_allow_html=True)

# ---------- chart helpers ----------
CLUSTER_LABELS = {
    "connectivity": "Connectivity", "room": "Room", "building": "Building",
    "food": "Food & Dining", "amenities": "Amenities", "service": "Service",
    "location": "Location", "parking": "Parking", "policies": "Policies",
    "value": "Value", "meta": "Accuracy",
}
CLUSTER_ORDER = list(CLUSTER_LABELS.keys())

def make_radar_chart(field_states: list[FieldState], topics, property_id: str) -> go.Figure:
    """Build a radar chart showing % coverage per cluster."""
    # Group field states by cluster
    cluster_known: dict[str, int] = {c: 0 for c in CLUSTER_ORDER}
    cluster_total: dict[str, int] = {c: 0 for c in CLUSTER_ORDER}

    fc = build_field_cluster_map(topics)
    for fs in field_states:
        if fs.eg_property_id != property_id:
            continue
        cluster = fc.get(fs.field_id, "")
        if cluster in cluster_total:
            cluster_total[cluster] += 1
            if fs.value_known:
                cluster_known[cluster] += 1

    labels = [CLUSTER_LABELS[c] for c in CLUSTER_ORDER]
    values = [
        round(100 * cluster_known[c] / cluster_total[c]) if cluster_total[c] > 0 else 0
        for c in CLUSTER_ORDER
    ]
    # Close the polygon
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=labels_closed,
        fill='toself',
        line=dict(color='#6366f1', width=2),
        fillcolor='rgba(99,102,241,0.15)',
        hovertemplate='%{theta}: %{r}%<extra></extra>',
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix='%', tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=30, b=30),
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def make_donut_chart(known: int, total: int) -> go.Figure:
    """Build a donut chart showing overall coverage."""
    pct = round(100 * known / total) if total else 0
    fig = go.Figure(data=[go.Pie(
        values=[known, total - known],
        labels=['Known', 'Unknown'],
        hole=0.75,
        marker=dict(colors=['#16a34a', '#e5e7eb']),
        textinfo='none',
        hovertemplate='%{label}: %{value}<extra></extra>',
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=180,
        paper_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(
            text=f'<b>{pct}%</b>', x=0.5, y=0.5,
            font_size=32, font_color='#00355F', showarrow=False,
        )],
    )
    return fig

def field_status(fs: FieldState | None, today: date) -> tuple[str, str]:
    """Return (label, css_class) for a field state."""
    if not fs or not fs.value_known:
        return ("Missing", "pill-missing")
    if (fs.short_ema is not None and fs.long_ema is not None
            and abs(fs.short_ema - fs.long_ema) >= 0.3):
        return ("Drifting", "pill-drifting")
    if fs.last_confirmed_date:
        age = (today - fs.last_confirmed_date).days
        if age > 90:
            return ("Stale", "pill-stale")
    return ("Known", "pill-known")

def score_bar_html(sf: ScoredField | None) -> str:
    """Render a tiny stacked bar showing missing/stale/coverage proportions."""
    if sf is None:
        return ""
    total = sf.missing + sf.stale + sf.coverage
    if total == 0:
        return "<div class='score-bar-wrap'><div class='score-bar'></div></div>"
    m_pct = 100 * sf.missing / total
    s_pct = 100 * sf.stale / total
    c_pct = 100 * sf.coverage / total
    return (
        f"<div class='score-bar-wrap'>"
        f"<div class='score-bar'>"
        f"<div class='seg-m' style='width:{m_pct:.0f}%'></div>"
        f"<div class='seg-s' style='width:{s_pct:.0f}%'></div>"
        f"<div class='seg-c' style='width:{c_pct:.0f}%'></div>"
        f"</div></div>"
    )

def _topic_display(fs):
    if not fs or not fs.value_known:
        return ("No mentions", "pill-missing")
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

# ---------- data ----------
repo = get_repo()
flow = get_flow()
topics = get_taxonomy()
today = _max_review_date(repo)

_topic_clusters: dict[str, list] = {}
for _t in topics:
    _topic_clusters.setdefault(_t.cluster_id, []).append(_t)

properties = repo.list_properties()
_property_id_list = [p.eg_property_id for p in properties]

def _format_property(pid: str) -> str:
    p = repo.get_property(pid)
    n = len(repo.list_reviews_for(pid))
    fs_list = repo.list_field_states_for(pid)
    known = sum(1 for fs in fs_list if fs.value_known)
    total = len(fs_list)
    pct = round(100 * known / total) if total else 0
    return f"{p.city or '?'}, {p.country or '?'}  ({p.star_rating or '?'}★ · {n} reviews · {pct}% covered)"

_has_unanswered = any(not a for _, a in st.session_state.get("pending_questions", []))
_active = st.session_state.get("active_property_id")
if _has_unanswered and _active and _active in _property_id_list:
    st.session_state.property_picker = _active

# ============================================================
# HEADER
# ============================================================
st.markdown(
    "<div class='header-bar'>"
    "<div><h1>Ask What Matters</h1>"
    "<p>Adaptive follow-up questions that fill missing property information</p></div>"
    "</div>",
    unsafe_allow_html=True,
)

hcol1, hcol2 = st.columns([5, 1])
with hcol1:
    property_id = st.selectbox(
        "Property", _property_id_list,
        format_func=_format_property,
        key="property_picker",
        disabled=_has_unanswered,
        label_visibility="collapsed",
    )
with hcol2:
    if st.button("Reset", help="Clear this session's answers"):
        for k in ["pending_questions", "answered_fields", "last_flashed_field",
                   "review_text", "active_property_id", "submit_result"]:
            st.session_state.pop(k, None)
        _init_state()
        st.rerun()

prop = repo.get_property(property_id)

# ============================================================
# TWO-COLUMN LAYOUT
# ============================================================
left, right = st.columns([2, 3], gap="large")

# ---------- LEFT: PROPERTY KNOWLEDGE PANEL ----------
with left:
    # Property card
    stars_html = f"<span class='star-pill'>{prop.star_rating or '?'}★</span>" if prop.star_rating else ""
    checkin = ""
    if prop.check_in_start_time:
        checkin = f"<br><span style='font-size:12px;color:#94a3b8;'>Check-in {prop.check_in_start_time}–{prop.check_in_end_time or '?'} · Out {prop.check_out_time or '?'}</span>"
    st.markdown(
        f"<div class='prop-card'>"
        f"<div class='city'>{prop.city or '?'}, {prop.country or '?'} {stars_html}</div>"
        f"<div class='meta'>Guest avg: {prop.guestrating_avg_expedia or '?'}/10{checkin}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Coverage donut + radar side by side
    all_fs = repo.list_field_states_for(property_id)
    known_count = sum(1 for fs in all_fs if fs.value_known)
    total_count = len(all_fs)

    donut_col, radar_col = st.columns([1, 2])
    with donut_col:
        st.markdown("<div class='coverage-label'>Coverage</div>", unsafe_allow_html=True)
        st.plotly_chart(make_donut_chart(known_count, total_count), use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"<div class='coverage-label'>{known_count} of {total_count} fields</div>", unsafe_allow_html=True)
    with radar_col:
        st.plotly_chart(make_radar_chart(all_fs, topics, property_id), use_container_width=True, config={'displayModeBar': False})

    # Build a lookup from field_id -> ScoredField for score bars
    result: SubmitResult | None = st.session_state.get("submit_result")
    sf_lookup: dict[str, ScoredField] = {}
    if result:
        for sf in result.scored_fields:
            sf_lookup[sf.field_state.field_id] = sf

    flash_field = st.session_state.last_flashed_field

    # --- Sub-ratings ---
    st.markdown("<div class='section-hdr'>Sub-Ratings</div>", unsafe_allow_html=True)
    for key in SUB_RATING_KEYS:
        fid = f"rating:{key}"
        fs = repo.get_field_state(property_id, fid)
        pretty = key.replace("_", " ").title()
        label, pill_cls = field_status(fs, today)
        css = "flash" if flash_field == fid else ""
        val = ""
        if fs and fs.value_known and fs.short_ema is not None:
            val = f" · {fs.short_ema:.1f}/5"
        bar = score_bar_html(sf_lookup.get(fid))
        st.markdown(
            f"<div class='field-row {css}'>"
            f"<span>{pretty}</span>"
            f"<span style='display:flex;align-items:center;gap:8px;'>"
            f"{bar}<span class='pill {pill_cls}'>{label}{val}</span>"
            f"</span></div>",
            unsafe_allow_html=True,
        )

    # --- Schema fields ---
    st.markdown("<div class='section-hdr'>Description Fields</div>", unsafe_allow_html=True)
    for sf_name in SCHEMA_DESCRIPTION_FIELDS:
        fid = f"schema:{sf_name}"
        fs = repo.get_field_state(property_id, fid)
        pretty = sf_name.replace("property_amenity_", "").replace("_", " ").title()
        label, pill_cls = field_status(fs, today)
        css = "flash" if flash_field == fid else ""
        bar = score_bar_html(sf_lookup.get(fid))
        st.markdown(
            f"<div class='field-row {css}'>"
            f"<span>{pretty}</span>"
            f"<span style='display:flex;align-items:center;gap:8px;'>"
            f"{bar}<span class='pill {pill_cls}'>{label}</span>"
            f"</span></div>",
            unsafe_allow_html=True,
        )

    # --- Guest topics by cluster ---
    st.markdown("<div class='section-hdr'>Guest Topics</div>", unsafe_allow_html=True)
    for cluster_id, cluster_topics in _topic_clusters.items():
        cluster_states = []
        has_known = False
        for t in cluster_topics:
            fid = f"topic:{t.topic_id}"
            fs = repo.get_field_state(property_id, fid)
            cluster_states.append((t, fs))
            if fs and fs.value_known:
                has_known = True

        known_c = sum(1 for _, fs in cluster_states if fs and fs.value_known)
        total_c = len(cluster_states)
        cluster_label = CLUSTER_LABELS.get(cluster_id, cluster_id.title())

        with st.expander(f"{cluster_label} ({known_c}/{total_c})", expanded=has_known):
            for t, fs in cluster_states:
                fid = f"topic:{t.topic_id}"
                css = "flash" if flash_field == fid else ""
                label, pill_cls = field_status(fs, today)
                display, _ = _topic_display(fs)
                bar = score_bar_html(sf_lookup.get(fid))
                st.markdown(
                    f"<div class='field-row {css}'>"
                    f"<span>{t.label}</span>"
                    f"<span style='display:flex;align-items:center;gap:8px;'>"
                    f"{bar}<span class='pill {pill_cls}'>{display}</span>"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )

# ---------- RIGHT: REVIEW & QUESTION FLOW ----------
with right:
    st.markdown("<div class='section-hdr'>Write Your Review</div>", unsafe_allow_html=True)

    if MIC_AVAILABLE:
        voice_text = speech_to_text(
            language="en", start_prompt="🎤 Record review", stop_prompt="⏹ Stop",
            just_once=True, use_container_width=False, key="voice_review",
        )
        if voice_text:
            st.session_state.review_text = voice_text
    else:
        st.caption("Install `streamlit-mic-recorder` for voice input.")

    review_text = st.text_area(
        "Your review (or leave blank for cold-start demo)",
        value=st.session_state.review_text,
        key="review_input", height=100,
        label_visibility="collapsed",
        placeholder="Write your review here, or leave blank to see what information is missing...",
    )

    if st.button("Submit review", type="primary", key="submit_review"):
        with st.spinner("Analyzing review — detecting language, embedding, tagging topics, scoring gaps..."):
            result = flow.submit_review(property_id, review_text, today=today)
        st.session_state.pending_questions = [(q, False) for q in result.questions]
        st.session_state.submit_result = result
        st.session_state.review_text = review_text
        st.session_state.active_property_id = property_id
        st.session_state.pending_review_id = flow._pending_review_id.get(property_id)
        st.rerun()

    # --- Enrichment summary ---
    result = st.session_state.get("submit_result")
    if result and isinstance(result, SubmitResult):
        meta = result.enrichment
        lang_chip = f"<div class='enrich-chip'>Language: <b>{meta.lang}</b></div>"
        trans_chip = (
            f"<div class='enrich-chip translated'>Translated to English</div>"
            if meta.translated else
            f"<div class='enrich-chip'>Original: English</div>"
        )
        emb_chip = (
            f"<div class='enrich-chip embedding'>Embedding: <b>{meta.embedding_dim}-dim</b></div>"
            if meta.embedding_dim > 0 else
            f"<div class='enrich-chip embedding'>No embedding (empty review)</div>"
        )
        topic_chip = f"<div class='enrich-chip topics'>Topics tagged: <b>{meta.topics_tagged}/{meta.topics_total}</b></div>"
        st.markdown(
            f"<div class='enrich-strip'>{lang_chip}{trans_chip}{emb_chip}{topic_chip}</div>",
            unsafe_allow_html=True,
        )

    # --- Follow-up questions ---
    if st.session_state.pending_questions:
        st.markdown("<div class='section-hdr'>Follow-up Questions</div>", unsafe_allow_html=True)

        for idx, (q, answered) in enumerate(st.session_state.pending_questions):
            if answered:
                continue

            # Find the ScoredField for this question
            sf_match = None
            if result and isinstance(result, SubmitResult):
                for sf in result.scored_fields:
                    if sf.field_state.field_id == q.field_id:
                        sf_match = sf
                        break

            st.markdown(
                f"<div class='q-card'>"
                f"<div class='q-text'>{q.question_text}</div>"
                f"<div class='q-reason'>Why we asked: {q.reason}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Score breakdown expander
            if sf_match:
                with st.expander("Why this question?", expanded=False):
                    st.markdown(
                        f"<span class='rank-badge'>Rank #{sf_match.rank} of {result.total_fields}</span>"
                        f"<span class='cluster-badge'>{sf_match.cluster.title() or 'Unknown'}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='score-detail'>"
                        f"Score = <span class='val'>{sf_match.composite:.3f}</span><br>"
                        f"&nbsp;&nbsp;= 0.55 × missing(<span class='val'>{sf_match.missing:.2f}</span>)"
                        f" + 0.25 × stale(<span class='val'>{sf_match.stale:.2f}</span>)"
                        f" + 0.15 × coverage(<span class='val'>{sf_match.coverage:.2f}</span>)"
                        f" − 0.35 × redundancy(<span class='val'>{sf_match.redundancy:.2f}</span>)"
                        f" + 0.02"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Input widget
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
                if st.button("Submit answer", key=f"submit_{idx}", type="primary"):
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
                        st.warning("Couldn't parse that — recorded as unscorable.")
                    st.rerun()
            with bcol2:
                if st.button("Skip", key=f"skip_{idx}"):
                    active_pid = st.session_state.get("active_property_id") or property_id
                    flow.submit_answer(active_pid, q, None, today=today)
                    st.session_state.pending_questions[idx] = (q, True)
                    st.rerun()

        if all(a for _, a in st.session_state.pending_questions):
            st.session_state.active_property_id = None
            st.success("All follow-ups addressed — submit another review to continue.")
```

**Step 2: Run the app and visually verify**

Run: `cd question-engine && streamlit run app/streamlit_app.py`

Check:
- Header bar renders with navy gradient
- Property card shows city, stars, guest avg
- Donut chart shows coverage percentage
- Radar chart shows 11 cluster axes
- Field rows have color-coded pills (green/amber/red)
- Submit a review → enrichment strip appears
- Question cards show with score breakdown expander
- Answer a question → left panel updates, flash animation fires

**Step 3: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: redesign Streamlit UI with radar chart, score breakdowns, and enrichment strip"
```

---

### Task 7: Run full test suite and fix any regressions

**Step 1: Run all tests**

Run: `cd question-engine && python -m pytest tests/ -v`
Expected: all PASS

**Step 2: Fix any failures**

Address import errors or interface mismatches from the `submit_review` return type change.

**Step 3: Commit if any fixes**

```bash
git add -A
git commit -m "fix: resolve test regressions from SubmitResult refactor"
```

---

### Task 8: Visual QA in browser

**Step 1: Start the dev server**

Run: `cd question-engine && streamlit run app/streamlit_app.py`

**Step 2: Test the golden path**

1. Select a property → verify donut + radar + field inventory render
2. Submit an empty review (cold start) → verify enrichment strip + questions appear
3. Open "Why this question?" → verify score breakdown shows
4. Answer Q1 → verify left panel updates (pill changes color, radar shifts)
5. Skip Q2 → verify completion message
6. Switch to a different property → verify data reloads

**Step 3: Test edge cases**

1. Property with high coverage → radar mostly filled, fewer questions
2. Property with low coverage → radar sparse, 2 questions generated
3. Reset button → clears state, returns to fresh view
