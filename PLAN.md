# Ask What Matters — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a prototype that, when a traveler submits a property review, asks 1–2 low-friction follow-up questions targeted at the specific information Expedia is missing or has stale for that property — and visibly updates the property's information when answered.

**Architecture:** Three-layer state model — (A) Expedia schema fields from `Description_PROC.csv` + sub-ratings from `Reviews_PROC.csv`, (B) a fixed ~40-concept global hospitality taxonomy, (C) per-(property × field) aggregate signals (EMAs, mention counts, last-confirmed dates). A deterministic ranker picks fields to ask about, weighting *missing* (value unknown) and *stale* (old or drifting) over *redundant* (already covered by the current review). An LLM renders the actual question from the picked field and the property's current state. A single SQLite file holds everything; a Streamlit app is the UI with Web Speech API for voice.

**Tech Stack:** Python 3.11+, OpenAI API (`gpt-4.1-mini` + `text-embedding-3-small`), Pydantic v2, SQLite (stdlib `sqlite3`), pandas for CSV ingest, numpy for embedding math, Streamlit for UI, Web Speech API (browser) for voice, `python-dotenv` for API key management, `pytest` for tests, `pyyaml` for config, `langdetect` for language sniffing.

**Repo layout (final):**
```
question-engine/
├── pyproject.toml
├── run.py                          # CLI: build | ask | serve
├── .env.example                    # placeholder for OPENAI_API_KEY
├── .gitignore                      # must exclude .env, .venv, data/state.sqlite, __pycache__
├── config/
│   ├── weights.yaml                # ranker weights + EMA half-lives
│   └── taxonomy.yaml               # fixed taxonomy of ~40 topics
├── data/
│   ├── state.sqlite                # built artifact (git-ignored)
│   └── cache/                      # LLM response cache (git-ignored)
├── src/
│   ├── __init__.py
│   ├── question_engine.py          # orchestrator
│   ├── models.py                   # Pydantic models
│   ├── db.py                       # SQLite repository
│   ├── ingest.py                   # CSV loading + cleaning
│   ├── llm.py                      # OpenAI client wrapper + caching
│   ├── embeddings.py               # embedding helpers + similarity
│   ├── taxonomy.py                 # taxonomy loading
│   ├── enrich.py                   # translate / tag / embed reviews
│   ├── signals.py                  # EMA + aggregation
│   ├── scoring.py                  # missing/stale/coverage scores
│   ├── ranker.py                   # combined rank + pick K
│   ├── renderer.py                 # LLM question-render
│   ├── parser.py                   # LLM answer-parse
│   └── flow.py                     # live ask-flow orchestrator
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── descriptions_tiny.csv
│   │   └── reviews_tiny.csv
│   ├── test_ingest.py
│   ├── test_signals.py
│   ├── test_scoring.py
│   ├── test_ranker.py
│   └── test_flow.py
├── app/
│   └── streamlit_app.py            # UI
└── README.md
```

**Global conventions:**
- All file paths in this plan are relative to the repo root (`/Users/sachabaniassad/Desktop/projects/Expedia Hackathon/`).
- Every phase ends with a green `pytest` run and a git commit.
- `make sure old test still pass` after each change — any new test must be additive.
- Never commit `.env`. Pre-commit grep for `sk-` before any push.
- All OpenAI calls go through `src/llm.py`; every call is SHA256-cached to `data/cache/`, so reruns are free.
- All times use the `acquisition_date` field (not wall-clock) so the demo is reproducible from static data. "Today" for staleness = max `acquisition_date` across the dataset.

---

## Phase 1 — Project scaffolding & dependencies

Set up the project so every later phase can import, test, and run.

### Task 1.1: Fill in dependencies

**Files:**
- Modify: `question-engine/pyproject.toml`

**Step 1: Replace the empty `dependencies = []` with the real list.**

```toml
[project]
name = "question-engine"
version = "0.1.0"
description = "Ask What Matters — adaptive follow-up question engine for Expedia reviews"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.30",
    "pydantic>=2.6",
    "pandas>=2.2",
    "numpy>=1.26",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "langdetect>=1.0.9",
    "streamlit>=1.33",
    "tenacity>=8.2",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=4.1", "ruff>=0.4"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

**Step 2: Install them.**

Run: `cd question-engine && python -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'`
Expected: no errors, all deps resolve.

**Step 3: Commit.**

```bash
git add question-engine/pyproject.toml
git commit -m "chore: add runtime and dev dependencies"
```

### Task 1.2: Env & gitignore

**Files:**
- Create: `question-engine/.env.example`
- Create: `question-engine/.gitignore`

**Step 1: Write `.env.example`:**

```
OPENAI_API_KEY=sk-REPLACE-ME
OPENAI_MODEL_CHAT=gpt-4.1-mini
OPENAI_MODEL_EMBED=text-embedding-3-small
```

**Step 2: Write `.gitignore`:**

```
.venv/
__pycache__/
*.pyc
.env
data/state.sqlite
data/cache/
.pytest_cache/
.coverage
```

**Step 3: Commit.**

```bash
git add question-engine/.env.example question-engine/.gitignore
git commit -m "chore: env template and gitignore"
```

### Task 1.3: Package skeleton

**Files:**
- Create: `question-engine/src/__init__.py` (empty)
- Create: `question-engine/tests/__init__.py` (empty)
- Create: `question-engine/tests/conftest.py`
- Create: `question-engine/data/` (directory, empty, with a `.gitkeep`)
- Create: `question-engine/config/` (directory)

**Step 1: Write `tests/conftest.py`:**

```python
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCES = REPO_ROOT / "hackathon resources"

@pytest.fixture(scope="session")
def descriptions_csv():
    return RESOURCES / "Description_PROC.csv"

@pytest.fixture(scope="session")
def reviews_csv():
    return RESOURCES / "Reviews_PROC.csv"

@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "state.sqlite"
```

**Step 2: Run `pytest` to confirm it collects zero tests without errors.**

Run: `cd question-engine && pytest`
Expected: `no tests ran in 0.XXs`

**Step 3: Commit.**

```bash
git add question-engine/src/__init__.py question-engine/tests/__init__.py question-engine/tests/conftest.py question-engine/data/.gitkeep question-engine/config/
git commit -m "chore: package skeleton"
```

---

## Phase 2 — Pydantic models & SQLite schema

Define the shape of every piece of state. Everything downstream depends on these types.

### Task 2.1: Pydantic models

**Files:**
- Create: `question-engine/src/models.py`
- Create: `question-engine/tests/test_models.py`

**Step 1: Write the failing test first.**

```python
# tests/test_models.py
from datetime import date
from src.models import Property, Review, FieldState, RatingBreakdown, TaxonomyTopic

def test_rating_breakdown_zero_is_none():
    r = RatingBreakdown.from_raw({"overall": 5, "checkin": 0, "service": 4})
    assert r.overall == 5
    assert r.checkin is None          # 0 in data means NULL
    assert r.service == 4

def test_property_accepts_missing_star_rating():
    p = Property(eg_property_id="abc", city="Pompei", country="Italy",
                 star_rating=None, guestrating_avg_expedia=8.4)
    assert p.star_rating is None

def test_field_state_ema_can_be_none_when_sparse():
    fs = FieldState(eg_property_id="abc", field_id="topic:wifi",
                    value_known=True, mention_count=2)
    assert fs.short_ema is None
    assert fs.long_ema is None
```

**Step 2: Run to see it fail.**

Run: `pytest tests/test_models.py -v`
Expected: `ModuleNotFoundError` on `src.models`.

**Step 3: Implement `src/models.py`:**

```python
from __future__ import annotations
from datetime import date
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field

SUB_RATING_KEYS = (
    "overall", "roomcleanliness", "service", "roomcomfort", "hotelcondition",
    "roomquality", "convenienceoflocation", "neighborhoodsatisfaction",
    "valueformoney", "roomamenitiesscore", "communication", "ecofriendliness",
    "checkin", "onlinelisting", "location",
)

class RatingBreakdown(BaseModel):
    overall: Optional[int] = None
    roomcleanliness: Optional[int] = None
    service: Optional[int] = None
    roomcomfort: Optional[int] = None
    hotelcondition: Optional[int] = None
    roomquality: Optional[int] = None
    convenienceoflocation: Optional[int] = None
    neighborhoodsatisfaction: Optional[int] = None
    valueformoney: Optional[int] = None
    roomamenitiesscore: Optional[int] = None
    communication: Optional[int] = None
    ecofriendliness: Optional[int] = None
    checkin: Optional[int] = None
    onlinelisting: Optional[int] = None
    location: Optional[int] = None

    @classmethod
    def from_raw(cls, d: dict | None) -> "RatingBreakdown":
        d = d or {}
        cleaned = {k: (v if isinstance(v, int) and v > 0 else None)
                   for k, v in d.items() if k in SUB_RATING_KEYS}
        return cls(**cleaned)

class Property(BaseModel):
    eg_property_id: str
    city: Optional[str] = None
    province: Optional[str] = None
    country: Optional[str] = None
    star_rating: Optional[float] = None
    guestrating_avg_expedia: Optional[float] = None
    area_description: Optional[str] = None
    property_description: Optional[str] = None
    amenities: dict[str, list[str]] = Field(default_factory=dict)  # subcategory -> items
    popular_amenities_list: list[str] = Field(default_factory=list)
    check_in_start_time: Optional[str] = None
    check_in_end_time: Optional[str] = None
    check_out_time: Optional[str] = None
    check_out_policy: Optional[str] = None
    pet_policy: Optional[str] = None
    children_and_extra_bed_policy: Optional[str] = None
    check_in_instructions: Optional[str] = None
    know_before_you_go: Optional[str] = None

class Review(BaseModel):
    review_id: str                        # synthetic: f"{property_id}:{row_index}"
    eg_property_id: str
    acquisition_date: date
    lob: Optional[str] = None
    rating: RatingBreakdown
    review_title: Optional[str] = None
    review_text_orig: Optional[str] = None
    review_text_en: Optional[str] = None  # filled by enrichment
    lang: Optional[str] = None
    source: Literal["csv", "live"] = "csv"

class TaxonomyTopic(BaseModel):
    topic_id: str                         # slug, e.g. "wifi"
    label: str                            # display, e.g. "WiFi"
    cluster_id: str                       # coarse group, e.g. "connectivity"
    question_hint: str                    # one-line hint the renderer uses

class FieldState(BaseModel):
    eg_property_id: str
    field_id: str                         # "schema:pet_policy" or "topic:wifi" or "rating:checkin"
    value_known: bool = False
    last_confirmed_date: Optional[date] = None
    short_ema: Optional[float] = None     # sentiment in [-1, 1] or rating mean in [1,5]
    long_ema: Optional[float] = None
    mention_count: int = 0
    last_asked_date: Optional[date] = None

class Question(BaseModel):
    field_id: str
    question_text: str
    input_type: Literal["rating_1_5", "yes_no", "short_text"]
    reason: str                           # "why we asked" — shown in UI

class Answer(BaseModel):
    field_id: str
    question_text: str
    answer_text: str
    parsed_value: Optional[str | int | float] = None
    status: Literal["scored", "unscorable", "skipped"]
```

**Step 4: Run the tests.**

Run: `pytest tests/test_models.py -v`
Expected: 3 passed.

**Step 5: Commit.**

```bash
git add question-engine/src/models.py question-engine/tests/test_models.py
git commit -m "feat: Pydantic models for properties, reviews, field state"
```

### Task 2.2: SQLite repository

**Files:**
- Create: `question-engine/src/db.py`
- Create: `question-engine/tests/test_db.py`

**Step 1: Write the test (roundtrip one property, one review, one field state).**

```python
# tests/test_db.py
from datetime import date
from src.db import Repo
from src.models import Property, Review, RatingBreakdown, FieldState

def test_roundtrip(tmp_db):
    repo = Repo(tmp_db)
    repo.init_schema()

    p = Property(eg_property_id="p1", city="Pompei", country="Italy",
                 guestrating_avg_expedia=8.4)
    repo.upsert_property(p)

    r = Review(review_id="p1:0", eg_property_id="p1",
               acquisition_date=date(2025, 9, 1),
               rating=RatingBreakdown(overall=5),
               review_text_orig="great")
    repo.upsert_review(r)

    fs = FieldState(eg_property_id="p1", field_id="rating:overall",
                    value_known=True, mention_count=1)
    repo.upsert_field_state(fs)

    assert repo.get_property("p1").city == "Pompei"
    assert repo.list_reviews_for("p1")[0].rating.overall == 5
    assert repo.get_field_state("p1", "rating:overall").mention_count == 1
```

**Step 2: Run to see it fail.**

Run: `pytest tests/test_db.py -v`
Expected: import error.

**Step 3: Implement `src/db.py`.**

The schema:
```sql
CREATE TABLE properties (
    eg_property_id TEXT PRIMARY KEY,
    raw_json TEXT NOT NULL
);
CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    eg_property_id TEXT NOT NULL,
    acquisition_date TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    embedding BLOB
);
CREATE INDEX idx_reviews_prop ON reviews(eg_property_id);
CREATE TABLE review_tags (
    review_id TEXT NOT NULL,
    field_id TEXT NOT NULL,
    mentioned INTEGER NOT NULL,
    sentiment INTEGER,                 -- -1 / 0 / 1 / NULL
    assertion TEXT,
    PRIMARY KEY (review_id, field_id)
);
CREATE TABLE field_state (
    eg_property_id TEXT NOT NULL,
    field_id TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    PRIMARY KEY (eg_property_id, field_id)
);
CREATE TABLE taxonomy (
    topic_id TEXT PRIMARY KEY,
    raw_json TEXT NOT NULL,
    embedding BLOB
);
CREATE TABLE answers (
    answer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id TEXT NOT NULL,
    field_id TEXT NOT NULL,
    raw_json TEXT NOT NULL
);
CREATE TABLE llm_cache (
    key TEXT PRIMARY KEY,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

Implementation pattern — serialize Pydantic to JSON for `raw_json` columns; store embeddings as raw `numpy.ndarray.tobytes()` with a fixed `np.float32` dtype.

```python
import sqlite3, json
from pathlib import Path
from datetime import date
import numpy as np
from src.models import Property, Review, FieldState, RatingBreakdown

class Repo:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row

    def init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)  # SCHEMA_SQL = the DDL above

    def upsert_property(self, p: Property) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO properties(eg_property_id, raw_json) VALUES (?, ?)",
                (p.eg_property_id, p.model_dump_json()))

    def upsert_review(self, r: Review, embedding: np.ndarray | None = None) -> None:
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO reviews(review_id, eg_property_id, acquisition_date, raw_json, embedding) VALUES (?, ?, ?, ?, ?)",
                (r.review_id, r.eg_property_id, r.acquisition_date.isoformat(),
                 r.model_dump_json(), emb_blob))

    def upsert_field_state(self, fs: FieldState) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO field_state(eg_property_id, field_id, raw_json) VALUES (?, ?, ?)",
                (fs.eg_property_id, fs.field_id, fs.model_dump_json()))

    def get_property(self, pid: str) -> Property | None:
        row = self._conn.execute(
            "SELECT raw_json FROM properties WHERE eg_property_id = ?", (pid,)).fetchone()
        return Property.model_validate_json(row["raw_json"]) if row else None

    def list_reviews_for(self, pid: str) -> list[Review]:
        rows = self._conn.execute(
            "SELECT raw_json FROM reviews WHERE eg_property_id = ? ORDER BY acquisition_date",
            (pid,)).fetchall()
        return [Review.model_validate_json(r["raw_json"]) for r in rows]

    def get_field_state(self, pid: str, fid: str) -> FieldState | None:
        row = self._conn.execute(
            "SELECT raw_json FROM field_state WHERE eg_property_id = ? AND field_id = ?",
            (pid, fid)).fetchone()
        return FieldState.model_validate_json(row["raw_json"]) if row else None
```

(Full file also includes: `list_properties`, `list_field_states_for`, `upsert_review_tags`, `record_answer`, `cache_get/cache_put`, `load_embedding`.)

**Step 4: Run tests.**

Run: `pytest tests/test_db.py -v`
Expected: pass.

**Step 5: Commit.**

```bash
git add question-engine/src/db.py question-engine/tests/test_db.py
git commit -m "feat: SQLite repository with Pydantic roundtrip"
```

---

## Phase 3 — CSV ingestion

Parse the two provided CSVs into the models. This phase is pure Python — no LLM yet — and is fully testable.

### Task 3.1: Parse Description_PROC.csv

**Files:**
- Create: `question-engine/src/ingest.py`
- Create: `question-engine/tests/fixtures/descriptions_tiny.csv` (5 rows cut from the real one)
- Create: `question-engine/tests/test_ingest.py`

**Step 1: Write the fixture.** Copy 5 rows of `Description_PROC.csv` with the full header into `tests/fixtures/descriptions_tiny.csv`.

**Step 2: Write the test.**

```python
# tests/test_ingest.py
from src.ingest import load_properties

def test_load_properties_parses_amenity_json():
    props = load_properties("tests/fixtures/descriptions_tiny.csv")
    assert len(props) == 5
    p0 = props[0]
    assert "internet" in p0.amenities
    assert any("wifi" in a.lower() for a in p0.amenities["internet"])

def test_load_properties_handles_missing_star_rating():
    props = load_properties("tests/fixtures/descriptions_tiny.csv")
    # at least one of the 5 fixtures has a NaN star_rating
    assert any(p.star_rating is None for p in props)
```

**Step 3: Implement `load_properties` in `src/ingest.py`:**

```python
import json, math, re
from pathlib import Path
import pandas as pd
from src.models import Property

AMENITY_COLS = [c for c in [
    "accessibility", "activities_nearby", "business_services", "conveniences",
    "family_friendly", "food_and_drink", "guest_services", "internet",
    "langs_spoken", "more", "outdoor", "parking", "spa", "things_to_do",
]]

def _parse_amenity_list(v):
    if not isinstance(v, str) or not v.strip():
        return []
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        # Some cells use single quotes; swap carefully.
        try: return json.loads(v.replace("'", '"'))
        except json.JSONDecodeError: return []

def _clean_text(v):
    if not isinstance(v, str): return None
    v = re.sub(r"\|MASK\|", "", v).strip()
    return v or None

def _nan_to_none(v):
    if v is None: return None
    if isinstance(v, float) and math.isnan(v): return None
    return v

def load_properties(path: str | Path) -> list[Property]:
    df = pd.read_csv(path)
    props = []
    for _, row in df.iterrows():
        amenities = {c: _parse_amenity_list(row.get(f"property_amenity_{c}"))
                     for c in AMENITY_COLS}
        props.append(Property(
            eg_property_id=row["eg_property_id"],
            city=_nan_to_none(row.get("city")),
            province=_nan_to_none(row.get("province")),
            country=_nan_to_none(row.get("country")),
            star_rating=_nan_to_none(row.get("star_rating")),
            guestrating_avg_expedia=_nan_to_none(row.get("guestrating_avg_expedia")),
            area_description=_clean_text(row.get("area_description")),
            property_description=_clean_text(row.get("property_description")),
            popular_amenities_list=_parse_amenity_list(row.get("popular_amenities_list")),
            amenities=amenities,
            check_in_start_time=_nan_to_none(row.get("check_in_start_time")),
            check_in_end_time=_nan_to_none(row.get("check_in_end_time")),
            check_out_time=_nan_to_none(row.get("check_out_time")),
            check_out_policy=_clean_text(row.get("check_out_policy")),
            pet_policy=_clean_text(row.get("pet_policy")),
            children_and_extra_bed_policy=_clean_text(row.get("children_and_extra_bed_policy")),
            check_in_instructions=_clean_text(row.get("check_in_instructions")),
            know_before_you_go=_clean_text(row.get("know_before_you_go")),
        ))
    return props
```

**Step 4: Run tests.**

Run: `pytest tests/test_ingest.py -v`
Expected: 2 passed.

**Step 5: Commit.**

```bash
git add question-engine/src/ingest.py question-engine/tests/fixtures/descriptions_tiny.csv question-engine/tests/test_ingest.py
git commit -m "feat: parse Description_PROC.csv into Property models"
```

### Task 3.2: Parse Reviews_PROC.csv

**Files:**
- Modify: `question-engine/src/ingest.py` (add `load_reviews`)
- Create: `question-engine/tests/fixtures/reviews_tiny.csv` (10 rows covering ≥2 properties and ≥2 languages)
- Modify: `question-engine/tests/test_ingest.py` (add review test)

**Step 1: Add failing test.**

```python
from datetime import date
from src.ingest import load_reviews

def test_load_reviews_parses_date_and_rating():
    reviews = load_reviews("tests/fixtures/reviews_tiny.csv")
    assert len(reviews) == 10
    r0 = reviews[0]
    assert isinstance(r0.acquisition_date, date)
    # 0 in rating JSON must become None
    for k in ("checkin", "location", "onlinelisting"):
        assert getattr(r0.rating, k) is None

def test_load_reviews_assigns_unique_ids():
    reviews = load_reviews("tests/fixtures/reviews_tiny.csv")
    assert len({r.review_id for r in reviews}) == len(reviews)
```

**Step 2: Implement.**

```python
from datetime import datetime
from src.models import Review, RatingBreakdown

def _parse_rating_json(v) -> dict:
    if not isinstance(v, str): return {}
    try: return json.loads(v)
    except json.JSONDecodeError:
        try: return json.loads(v.replace("'", '"'))
        except json.JSONDecodeError: return {}

def _parse_acquisition_date(v: str):
    # Format: M/D/YY (e.g. "9/9/25")
    return datetime.strptime(v.strip(), "%m/%d/%y").date()

def load_reviews(path: str | Path) -> list[Review]:
    df = pd.read_csv(path)
    reviews = []
    for i, row in df.iterrows():
        rating = RatingBreakdown.from_raw(_parse_rating_json(row.get("rating")))
        reviews.append(Review(
            review_id=f"{row['eg_property_id']}:{i}",
            eg_property_id=row["eg_property_id"],
            acquisition_date=_parse_acquisition_date(str(row["acquisition_date"])),
            lob=_nan_to_none(row.get("lob")),
            rating=rating,
            review_title=_clean_text(row.get("review_title")),
            review_text_orig=_clean_text(row.get("review_text")),
        ))
    return reviews
```

**Step 3: Run tests.**

Run: `pytest tests/test_ingest.py -v`
Expected: all 4 pass.

**Step 4: Commit.**

```bash
git add question-engine/src/ingest.py question-engine/tests/fixtures/reviews_tiny.csv question-engine/tests/test_ingest.py
git commit -m "feat: parse Reviews_PROC.csv into Review models"
```

---

## Phase 4 — Taxonomy & field registry

This is the fixed vocabulary of things we can ask about. No LLM calls here — the list is hand-curated and committed.

### Task 4.1: Write the taxonomy config

**Files:**
- Create: `question-engine/config/taxonomy.yaml`

**Step 1: Write `config/taxonomy.yaml`.** ~40 topics grouped into clusters. Each topic has `id`, `label`, `cluster`, `question_hint`:

```yaml
topics:
  # connectivity
  - { id: wifi,              label: "WiFi",                    cluster: connectivity, question_hint: "speed and coverage of WiFi" }
  - { id: cell_reception,    label: "Cell reception",          cluster: connectivity, question_hint: "mobile phone reception on-site" }
  # room
  - { id: room_cleanliness,  label: "Room cleanliness",        cluster: room,         question_hint: "how clean the room was" }
  - { id: bed_comfort,       label: "Bed comfort",             cluster: room,         question_hint: "mattress and bedding comfort" }
  - { id: shower_pressure,   label: "Shower / water pressure", cluster: room,         question_hint: "water pressure and temperature" }
  - { id: ac_heating,        label: "AC & heating",            cluster: room,         question_hint: "room temperature control" }
  - { id: noise,             label: "Noise level",             cluster: room,         question_hint: "sound from street/hallway" }
  # building
  - { id: elevator,          label: "Elevator",                cluster: building,     question_hint: "elevator availability and condition" }
  - { id: accessibility,     label: "Accessibility",           cluster: building,     question_hint: "step-free access, grab bars, ramps" }
  - { id: recent_renovation, label: "Recent renovation",       cluster: building,     question_hint: "whether property looks recently updated" }
  # food
  - { id: breakfast,         label: "Breakfast",               cluster: food,         question_hint: "breakfast availability and quality" }
  - { id: onsite_dining,     label: "On-site dining",          cluster: food,         question_hint: "hotel restaurant / bar" }
  # amenities
  - { id: pool,              label: "Pool",                    cluster: amenities,    question_hint: "pool availability and condition" }
  - { id: gym,               label: "Gym",                     cluster: amenities,    question_hint: "fitness center" }
  - { id: spa,               label: "Spa",                     cluster: amenities,    question_hint: "spa services" }
  - { id: business_center,   label: "Business center",         cluster: amenities,    question_hint: "workspace / printing / meeting rooms" }
  # service
  - { id: staff_helpfulness, label: "Staff helpfulness",       cluster: service,      question_hint: "how responsive and friendly staff were" }
  - { id: checkin_ease,      label: "Check-in ease",           cluster: service,      question_hint: "speed and smoothness of check-in" }
  - { id: language_barriers, label: "Language barriers",       cluster: service,      question_hint: "staff English / local language" }
  # location
  - { id: walkability,       label: "Walkability",             cluster: location,     question_hint: "walking access to attractions / transit" }
  - { id: neighborhood_safety, label: "Neighborhood safety",   cluster: location,     question_hint: "feeling safe walking around area" }
  - { id: transit_access,    label: "Transit access",          cluster: location,     question_hint: "proximity to public transit / airport shuttle" }
  # parking / access
  - { id: parking_availability, label: "Parking availability", cluster: parking,      question_hint: "parking presence, cost, and ease" }
  - { id: ev_charging,       label: "EV charging",             cluster: parking,      question_hint: "electric vehicle charging on-site" }
  # policies / stay
  - { id: pet_friendly,      label: "Pet policy in practice",  cluster: policies,     question_hint: "what it was actually like traveling with a pet" }
  - { id: family_friendly,   label: "Family friendliness",     cluster: policies,     question_hint: "amenities/services for children" }
  - { id: value_for_money,   label: "Value for money",         cluster: value,         question_hint: "felt price-to-quality ratio" }
  - { id: hidden_fees,       label: "Hidden fees",             cluster: value,         question_hint: "resort/destination/parking fees not listed" }
  # misc
  - { id: listing_accuracy,  label: "Listing accuracy",        cluster: meta,         question_hint: "whether photos/description matched reality" }
  - { id: safety_security,   label: "In-room security",        cluster: meta,         question_hint: "safe, door locks, in-room security" }
```

Target size: ~30 topics (enough to be useful, small enough to fit in one LLM prompt comfortably).

### Task 4.2: Load taxonomy + enumerate schema fields

**Files:**
- Create: `question-engine/src/taxonomy.py`
- Create: `question-engine/tests/test_taxonomy.py`

**Step 1: Write the test.**

```python
from src.taxonomy import load_taxonomy, schema_field_ids, all_field_ids

def test_load_taxonomy_has_topics():
    topics = load_taxonomy("config/taxonomy.yaml")
    assert len(topics) >= 20
    assert all(t.topic_id and t.label and t.cluster_id for t in topics)

def test_schema_fields_include_empty_subratings():
    fids = schema_field_ids()
    assert "rating:checkin" in fids
    assert "schema:pet_policy" in fids
    assert "schema:property_amenity_spa" in fids

def test_all_field_ids_is_union():
    assert set(all_field_ids()) == set(schema_field_ids()) | {f"topic:{t.topic_id}" for t in load_taxonomy("config/taxonomy.yaml")}
```

**Step 2: Implement `src/taxonomy.py`.**

```python
from pathlib import Path
import yaml
from src.models import TaxonomyTopic, SUB_RATING_KEYS

SCHEMA_DESCRIPTION_FIELDS = (
    "pet_policy", "children_and_extra_bed_policy", "check_in_instructions",
    "know_before_you_go", "check_out_policy",
    "property_amenity_accessibility", "property_amenity_activities_nearby",
    "property_amenity_business_services", "property_amenity_food_and_drink",
    "property_amenity_outdoor", "property_amenity_spa",
    "property_amenity_things_to_do",
)

def load_taxonomy(path: str | Path) -> list[TaxonomyTopic]:
    data = yaml.safe_load(Path(path).read_text())
    return [TaxonomyTopic(
        topic_id=t["id"], label=t["label"],
        cluster_id=t["cluster"], question_hint=t["question_hint"],
    ) for t in data["topics"]]

def schema_field_ids() -> list[str]:
    rating_ids = [f"rating:{k}" for k in SUB_RATING_KEYS]
    desc_ids = [f"schema:{f}" for f in SCHEMA_DESCRIPTION_FIELDS]
    return rating_ids + desc_ids

def all_field_ids(tax_path: str | Path = "config/taxonomy.yaml") -> list[str]:
    return schema_field_ids() + [f"topic:{t.topic_id}" for t in load_taxonomy(tax_path)]
```

**Step 3: Run tests.**

Run: `pytest tests/test_taxonomy.py -v`
Expected: 3 passed.

**Step 4: Commit.**

```bash
git add question-engine/config/taxonomy.yaml question-engine/src/taxonomy.py question-engine/tests/test_taxonomy.py
git commit -m "feat: hospitality taxonomy and schema field registry"
```

---

## Phase 5 — LLM wrapper & caching

One chokepoint for all OpenAI calls, with caching keyed by input hash so rebuilds cost $0.

### Task 5.1: `src/llm.py`

**Files:**
- Create: `question-engine/src/llm.py`
- Create: `question-engine/tests/test_llm.py`

**Step 1: Test (uses cache; does NOT hit the network).**

```python
from unittest.mock import patch, MagicMock
from src.llm import LlmClient

def test_cache_hit_skips_network(tmp_path):
    client = LlmClient(cache_dir=tmp_path, api_key="fake")
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content='{"x": 1}'))]
    with patch.object(client._client.chat.completions, "create", return_value=fake_resp) as m:
        a = client.chat_json(system="sys", user="u", model="gpt-4.1-mini")
        b = client.chat_json(system="sys", user="u", model="gpt-4.1-mini")
    assert a == b == {"x": 1}
    assert m.call_count == 1  # second call hit cache
```

**Step 2: Implement.**

```python
import hashlib, json, os
from pathlib import Path
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

class LlmClient:
    def __init__(self, cache_dir: str | Path = "data/cache",
                 api_key: str | None = None):
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def _key(self, **parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def _cache_get(self, key):
        p = self.cache / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _cache_put(self, key, value):
        (self.cache / f"{key}.json").write_text(json.dumps(value))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat_json(self, *, system: str, user: str, model: str,
                  temperature: float = 0.0) -> dict:
        key = self._key(kind="chat_json", system=system, user=user,
                        model=model, temperature=temperature)
        if (hit := self._cache_get(key)) is not None: return hit
        resp = self._client.chat.completions.create(
            model=model, temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        out = json.loads(resp.choices[0].message.content)
        self._cache_put(key, out)
        return out

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat_text(self, *, system: str, user: str, model: str,
                  temperature: float = 0.3) -> str:
        key = self._key(kind="chat_text", system=system, user=user,
                        model=model, temperature=temperature)
        if (hit := self._cache_get(key)) is not None: return hit["text"]
        resp = self._client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        text = resp.choices[0].message.content.strip()
        self._cache_put(key, {"text": text})
        return text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed(self, text: str, model: str = "text-embedding-3-small") -> list[float]:
        key = self._key(kind="embed", text=text, model=model)
        if (hit := self._cache_get(key)) is not None: return hit["vector"]
        resp = self._client.embeddings.create(model=model, input=[text])
        vec = resp.data[0].embedding
        self._cache_put(key, {"vector": vec})
        return vec
```

**Step 3: Run tests.**

Run: `pytest tests/test_llm.py -v`
Expected: 1 passed.

**Step 4: Commit.**

```bash
git add question-engine/src/llm.py question-engine/tests/test_llm.py
git commit -m "feat: LLM client with on-disk response cache"
```

---

## Phase 6 — Offline enrichment pipeline

Translate, embed, and tag every review with taxonomy topics + sentiment. This is the heavy LLM phase; run it once, cache everything.

### Task 6.1: Language detection

**Files:**
- Create: `question-engine/src/enrich.py` (partial)
- Create: `question-engine/tests/test_enrich.py`

**Step 1: Test.**

```python
from src.enrich import detect_language
def test_detect_language_basic():
    assert detect_language("The room was clean and staff friendly") == "en"
    assert detect_language("Das Zimmer war sehr sauber") == "de"
    assert detect_language("") == "unknown"
    assert detect_language(None) == "unknown"
```

**Step 2: Implement (inside `src/enrich.py`).**

```python
from langdetect import detect, DetectorFactory, LangDetectException
DetectorFactory.seed = 0  # deterministic

def detect_language(text: str | None) -> str:
    if not text or not text.strip(): return "unknown"
    try: return detect(text)
    except LangDetectException: return "unknown"
```

**Step 3: Run.**
Run: `pytest tests/test_enrich.py -v`

**Step 4: Commit.**

### Task 6.2: Translate non-English reviews

**Files:** Modify: `src/enrich.py`, `tests/test_enrich.py`.

Add `translate_to_english(text, lang, llm)` using `LlmClient.chat_text`. System prompt: "Translate to English. Preserve meaning. Do not add commentary." Mock the LLM in tests.

### Task 6.3: Tag reviews with taxonomy + sentiment

**Files:** Modify: `src/enrich.py`, `tests/test_enrich.py`.

`tag_review(review_en: str, topics: list[TaxonomyTopic], llm)` returns `dict[topic_id, {mentioned: bool, sentiment: -1|0|1|None, assertion: str|None}]`. One call per review, JSON mode. System prompt includes the full taxonomy labels + hints.

**User prompt template:**
```
Review: {text}

For each topic below, decide:
- mentioned: did the review talk about this topic? (true/false)
- sentiment: if mentioned, what was the sentiment? -1 (negative), 0 (neutral), 1 (positive), null (unclear)
- assertion: if the review stated a concrete fact about this topic, quote or paraphrase it in ≤15 words; else null

Return ONLY valid JSON: {"topic_id": {"mentioned": ..., "sentiment": ..., "assertion": ...}, ...}

Topics:
- wifi: speed and coverage of WiFi
- pool: pool availability and condition
...
```

Batched variant: `tag_reviews_batch(texts: list[str], topics, llm)` takes up to 10 reviews per call, returns a list of per-review dicts.

### Task 6.4: Build script — `run.py build`

**Files:**
- Modify: `question-engine/src/question_engine.py` (real orchestrator)
- Modify: `question-engine/run.py` (CLI dispatch)

`python run.py build` must:
1. Load `hackathon resources/Description_PROC.csv` → `load_properties` → upsert.
2. Load `hackathon resources/Reviews_PROC.csv` → `load_reviews`.
3. For each review: `detect_language`, `translate_to_english` (if needed), embed (text_en), tag (batched).
4. Upsert review + embedding + tags to DB.
5. Print progress: `[p_id] 120/152 tagged`.

**Verification (manual):**

Run: `cd question-engine && python run.py build`
Expected first-run: ~5–15 minutes depending on LLM rate limit (6k reviews ÷ 10/batch × 1–2s/call). Progress bar visible. `data/state.sqlite` ~30–60 MB. Second run completes in seconds (all cached).

**Sanity check after build:**
```bash
sqlite3 data/state.sqlite 'SELECT COUNT(*) FROM reviews;'     # 5999
sqlite3 data/state.sqlite 'SELECT COUNT(*) FROM review_tags;' # ~30 * 5999 (if all topics per review)
```

**Commit after each sub-task works.**

---

## Phase 7 — Signal aggregation

Roll `review_tags` into per-(property × field) `field_state` rows. No LLM calls.

### Task 7.1: EMA helpers

**Files:**
- Create: `question-engine/src/signals.py`
- Create: `question-engine/tests/test_signals.py`

**Step 1: Test.**

```python
from src.signals import ema_series

def test_ema_empty():
    assert ema_series([], half_life=5) is None

def test_ema_matches_closed_form():
    # constant stream of 1.0 → EMA converges to 1.0
    result = ema_series([1.0] * 100, half_life=5)
    assert abs(result - 1.0) < 1e-6

def test_ema_recency_weighted():
    # 10 negatives followed by 3 positives — short-half-life EMA should skew positive
    vals = [-1.0]*10 + [1.0]*3
    short = ema_series(vals, half_life=2)
    long = ema_series(vals, half_life=20)
    assert short > long
```

**Step 2: Implement.**

```python
import math
def ema_series(values: list[float], half_life: float) -> float | None:
    if not values: return None
    alpha = 1 - math.exp(-math.log(2) / half_life)
    s = values[0]
    for v in values[1:]:
        s = alpha * v + (1 - alpha) * s
    return s
```

**Step 3: Commit.**

### Task 7.2: Build field_state for sub-ratings

**Files:** Modify: `src/signals.py`, `tests/test_signals.py`.

`build_rating_field_states(reviews: list[Review]) -> list[FieldState]`: for each `(property, rating_key)`, collect the non-None sub-rating values over time, compute `short_ema` (half_life=5), `long_ema` (half_life=30), `mention_count`, `last_confirmed_date = max acquisition_date among non-None`. `value_known = mention_count > 0`. Emit `field_id=f"rating:{rating_key}"`.

**Test:**
```python
def test_rating_state_100pct_null_field_is_unknown(reviews_tiny):
    states = build_rating_field_states(reviews_tiny)
    checkin = next(s for s in states if s.field_id == "rating:checkin")
    assert checkin.value_known is False
    assert checkin.mention_count == 0
```

### Task 7.3: Build field_state for schema description fields

**Files:** Modify: `src/signals.py`, `tests/test_signals.py`.

`build_schema_field_states(properties)`: for each schema description field, `value_known = truthy(field_value)`. `last_confirmed_date = max_review_date_per_property` (assumption: description was valid as of the latest review — a documented limitation). No EMA (schema fields aren't sentiment-bearing).

### Task 7.4: Build field_state for taxonomy topics

**Files:** Modify: `src/signals.py`, `tests/test_signals.py`.

`build_topic_field_states(reviews, review_tags)`: for each `(property, topic)`, use tags where `mentioned=True` and `sentiment is not None`. Convert -1/0/1 to floats. Compute EMAs, `mention_count`, `last_confirmed_date = last mention date`.

### Task 7.5: Aggregation orchestrator

**Files:** Modify: `src/signals.py`.

`build_all_field_states(repo: Repo) -> None`: runs the three builders above and upserts everything. Called from `run.py build` after enrichment completes.

**Commit after tests pass.**

---

## Phase 8 — Gap scoring & ranker

Pure math, fully testable with no LLM.

### Task 8.1: Scoring functions

**Files:**
- Create: `question-engine/src/scoring.py`
- Create: `question-engine/tests/test_scoring.py`

**Tests first:**

```python
from datetime import date
from src.models import FieldState
from src.scoring import missing_score, stale_score, coverage_gap_score

def test_missing_score_unknown_is_1():
    fs = FieldState(eg_property_id="p", field_id="rating:checkin",
                    value_known=False, mention_count=0)
    assert missing_score(fs) == 1.0

def test_missing_score_known_is_0():
    fs = FieldState(eg_property_id="p", field_id="rating:overall",
                    value_known=True, mention_count=100)
    assert missing_score(fs) == 0.0

def test_stale_score_clips_at_1():
    fs = FieldState(eg_property_id="p", field_id="topic:wifi",
                    value_known=True, mention_count=20,
                    last_confirmed_date=date(2024, 1, 1),
                    short_ema=0.5, long_ema=0.5)
    today = date(2025, 9, 9)
    assert stale_score(fs, today=today) == 1.0

def test_stale_score_sentiment_drift_bonus():
    fs = FieldState(eg_property_id="p", field_id="topic:wifi",
                    value_known=True, mention_count=20,
                    last_confirmed_date=date(2025, 9, 1),
                    short_ema=-0.8, long_ema=0.8)   # delta = 1.6 > 0.5
    today = date(2025, 9, 9)
    assert stale_score(fs, today=today) >= 0.5     # drift term kicks in

def test_coverage_gap_decay():
    fs0 = FieldState(eg_property_id="p", field_id="topic:wifi", mention_count=0)
    fs10 = FieldState(eg_property_id="p", field_id="topic:wifi", mention_count=10)
    assert coverage_gap_score(fs0) > coverage_gap_score(fs10)
```

**Implementation:**

```python
from datetime import date

def missing_score(fs) -> float:
    return 0.0 if fs.value_known else 1.0

def stale_score(fs, today: date, time_horizon_days: int = 180) -> float:
    if not fs.value_known:
        return 0.0  # unknown is handled by missing_score, not here
    time_term = 0.0
    if fs.last_confirmed_date:
        age = (today - fs.last_confirmed_date).days
        time_term = max(0.0, min(1.0, age / time_horizon_days))
    drift_term = 0.0
    if (fs.short_ema is not None and fs.long_ema is not None
            and fs.mention_count >= 5
            and abs(fs.short_ema - fs.long_ema) > 0.5):
        drift_term = 0.5
    return min(1.0, time_term + drift_term)

def coverage_gap_score(fs) -> float:
    return 1.0 / (1.0 + fs.mention_count)
```

### Task 8.2: Ranker with weights config

**Files:**
- Create: `question-engine/config/weights.yaml`
- Create: `question-engine/src/ranker.py`
- Create: `question-engine/tests/test_ranker.py`

**`weights.yaml`:**
```yaml
w_missing: 0.55
w_stale: 0.25
w_coverage: 0.15
w_redundancy: 0.35
epsilon: 0.02
min_score_for_second_question: 0.40
short_ema_half_life: 5
long_ema_half_life: 30
time_horizon_days: 180
```

**Tests (examples):**
```python
def test_ranker_picks_missing_over_known():
    # Property with rating:checkin fully unknown vs rating:overall fully known
    # should pick rating:checkin.

def test_ranker_redundancy_penalizes_similar_topic():
    # Review embedding similar to topic:wifi → topic:wifi should be penalized
    # even if it otherwise ranked highest.

def test_ranker_respects_min_score_for_k2():
    # When the 2nd-best score is below min_score_for_second_question,
    # rank() returns only one question.

def test_ranker_dedupes_same_cluster():
    # If top 2 picks are from the same cluster (e.g. both "room"),
    # the 2nd is bumped to the next different-cluster candidate.
```

**Implementation sketch:**

```python
import numpy as np, yaml
from pathlib import Path
from src.models import FieldState, TaxonomyTopic
from src.scoring import missing_score, stale_score, coverage_gap_score

def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def rank_fields(
    *, property_id: str, field_states: list[FieldState],
    today, topic_embeddings: dict[str, np.ndarray],
    review_embedding: np.ndarray | None,
    field_cluster: dict[str, str],
    weights_path: str = "config/weights.yaml",
) -> list[tuple[FieldState, float]]:
    W = yaml.safe_load(Path(weights_path).read_text())
    scored = []
    for fs in field_states:
        if fs.eg_property_id != property_id: continue
        m = missing_score(fs)
        s = stale_score(fs, today=today, time_horizon_days=W["time_horizon_days"])
        c = coverage_gap_score(fs)
        red = 0.0
        if review_embedding is not None and fs.field_id.startswith("topic:"):
            topic_id = fs.field_id.split(":", 1)[1]
            if topic_id in topic_embeddings:
                red = max(0.0, _cosine(review_embedding, topic_embeddings[topic_id]))
        score = (W["w_missing"] * m + W["w_stale"] * s + W["w_coverage"] * c
                 - W["w_redundancy"] * red + W["epsilon"])
        scored.append((fs, score))
    return sorted(scored, key=lambda x: -x[1])

def pick_k(ranked: list[tuple[FieldState, float]],
           field_cluster: dict[str, str],
           min_score_for_k2: float = 0.40) -> list[FieldState]:
    if not ranked: return []
    picks = [ranked[0][0]]
    first_cluster = field_cluster.get(ranked[0][0].field_id)
    for fs, score in ranked[1:]:
        if score < min_score_for_k2: break
        if field_cluster.get(fs.field_id) == first_cluster: continue
        picks.append(fs); break
    return picks
```

`field_cluster` is a dict `{"topic:wifi": "connectivity", "rating:checkin": "service", "schema:pet_policy": "policies", …}` derived from the taxonomy + a small static map for schema fields.

**Commit after tests pass.**

---

## Phase 9 — Question rendering & answer parsing

LLM calls. One for rendering the question, one for parsing the answer.

### Task 9.1: Renderer

**Files:**
- Create: `question-engine/src/renderer.py`
- Create: `question-engine/tests/test_renderer.py`

**Test:** mock `LlmClient.chat_text` to return a fixed string, verify the prompt includes the field_id, current known value, and the question_hint.

**Implementation sketch:**

```python
def render_question(*, field_state: FieldState, property_: Property,
                    topic: TaxonomyTopic | None, llm) -> Question:
    # Determine input_type:
    if field_state.field_id.startswith("rating:"):
        input_type = "rating_1_5"
    elif field_state.field_id.startswith("schema:property_amenity_"):
        input_type = "yes_no"
    else:
        input_type = "short_text"

    # Build context block
    current = _current_value_for(field_state, property_)
    hint = topic.question_hint if topic else _schema_hint(field_state.field_id)
    cluster_context = f"Property: {property_.city}, {property_.country}, {property_.star_rating or '?'}-star"

    sys = ("You write ONE short follow-up question (≤20 words) for a hotel reviewer. "
           "Plain language. No compound questions. No pleasantries. Output ONLY the question.")
    usr = (f"{cluster_context}\n"
           f"Topic: {hint}\n"
           f"Current knowledge: {current}\n"
           f"Required answer type: {input_type}\n"
           f"Question:")
    text = llm.chat_text(system=sys, user=usr, model="gpt-4.1-mini", temperature=0.3)

    reason = _reason_for(field_state)  # "Check-in has no rating data for this property."
    return Question(field_id=field_state.field_id, question_text=text,
                    input_type=input_type, reason=reason)
```

`_current_value_for` returns `"unknown"` if the field is missing, else a concise stringified value (e.g., `"pet_policy says 'no pets allowed'"`, `"1,026 reviews, rated 4.6/5 overall but roomcomfort never rated"`).

### Task 9.2: Answer parser

**Files:**
- Create: `question-engine/src/parser.py`
- Create: `question-engine/tests/test_parser.py`

`parse_answer(question: Question, answer_text: str, llm) -> Answer`:
- If `input_type == rating_1_5`: try regex first (e.g., `r"([1-5])"`); fall back to `chat_json` with schema `{rating: int 1-5 or null, abstain: bool}`.
- If `yes_no`: regex for yes/no patterns; fall back to LLM.
- If `short_text`: LLM call returns `{value: str|null, abstain: bool}`. If answer is <3 chars or purely filler ("ok", "idk"), mark as `unscorable`.

**Output:**
- `status = "scored"` with `parsed_value` set, or
- `status = "unscorable"` with `parsed_value = None`, or
- `status = "skipped"` (when the UI sends a skip signal).

**Commit.**

---

## Phase 10 — Live flow orchestrator

Glue everything together. This is what the UI calls.

### Task 10.1: `src/flow.py`

**Files:**
- Create: `question-engine/src/flow.py`
- Create: `question-engine/tests/test_flow.py`

Public API:

```python
class AskFlow:
    def __init__(self, repo: Repo, llm: LlmClient,
                 taxonomy: list[TaxonomyTopic],
                 topic_embeddings: dict[str, np.ndarray],
                 field_cluster: dict[str, str],
                 weights_path: str = "config/weights.yaml"): ...

    def submit_review(self, property_id: str, review_text: str,
                      today: date) -> list[Question]:
        """
        1. detect + translate if needed
        2. embed text_en
        3. classify topics + sentiments (single LLM call)
        4. apply classification to in-memory field_state (does NOT persist yet —
           we wait until the follow-up is answered, so a skip doesn't pollute data)
        5. rank + pick K
        6. render each picked question
        """

    def submit_answer(self, property_id: str, question: Question,
                      answer_text: str, today: date) -> Answer:
        """
        1. parse answer
        2. update field_state for the picked field (value_known, EMA, mention_count,
           last_confirmed_date, last_asked_date)
        3. persist the original review + tags + the new answer (so the pending
           review from submit_review becomes durable only here)
        4. return the parsed Answer
        """
```

**Test strategy:** integration test with mocked LLM, a tiny fixture DB seeded from the 10-row review fixture. Assert that:
- A property with `rating:checkin` fully empty gets a `rating:checkin` follow-up.
- After answering that question, `field_state` for `rating:checkin` shows `value_known=True, mention_count=1, last_confirmed_date=today`.
- Submitting a second review similar to the just-asked field gets a *different* field (redundancy penalty).

**Commit.**

### Task 10.2: CLI dry-run

**Files:** Modify: `question-engine/run.py` to add `python run.py ask <property_id> "<review text>"`.

Prints the picked questions with their `reason`, accepts typed answers, shows the updated state. Lets us verify end-to-end without a browser.

**Verification:**

```bash
python run.py build                                               # once
python run.py ask 110f01b8ae518a... "Great location, loved it."
# Expected output shape:
#   Q1 (rating:checkin) — "How smooth was your check-in at the hotel?" [1-5]
#   Reason: Check-in has no rating data for this property.
#   Your answer: _
```

**Commit.**

---

## Phase 11 — Streamlit UI

The visible part. Judges will see this.

### Task 11.1: Basic layout

**Files:**
- Create: `question-engine/app/streamlit_app.py`

Sections:
1. **Top bar**: property selector (dropdown of the 13 properties, labeled "City, Country (Star★, N reviews)"), plus a "Reset demo" button that clears session answers without touching the DB.
2. **Property info card** (left column, sticky): top-level facts (star, city, guest rating, check-in window) + a list of sub-ratings with filled stars for populated ones and grey `—` for unrated ones + a small amenity coverage bar.
3. **Write your review** (right column): `st.text_area`, mic button (HTML/JS component using `SpeechRecognition`), Submit button.
4. **Follow-up card** (appears after submit): one or two question cards, each with `question_text` (big), `reason` (small), input widget matching `input_type`, Skip, Submit-answer, mic button.
5. **Update animation**: when an answer is submitted, the info card on the left re-renders with the newly filled field highlighted (a yellow flash via a small CSS class).
6. **Coverage meter** (top-right): "Known fields: 34 / 52" updating live.

**Voice implementation note:** Streamlit can host a small HTML+JS block via `components.v1.html`. That block uses the Web Speech API to capture audio → transcribe → POST the text back into `st.session_state` via `streamlit-js-eval` or a hidden form. No backend STT needed.

### Task 11.2: Wire up `AskFlow`

On "Submit review": call `flow.submit_review(property_id, text, today=MAX_REVIEW_DATE)`. Render the returned questions. On "Submit answer": call `flow.submit_answer(...)`, then rerun to refresh the info card.

### Task 11.3: Manual verification checklist

Run: `cd question-engine && streamlit run app/streamlit_app.py`

Check each of the following by hand:
- [ ] Picking a property shows that property's data (city, star, sub-ratings, amenity gaps).
- [ ] A high-volume property (e.g., Bell Gardens, 1094 reviews) picks something other than `rating:overall` as its top gap.
- [ ] A cold-start property (e.g., Freudenstadt, 8 reviews) picks a schema field the description is missing (e.g., `schema:property_amenity_spa`).
- [ ] Writing "The WiFi was awful" bumps WiFi's redundancy penalty and the picked question is NOT about WiFi.
- [ ] Writing an empty review still returns exactly 1 question (cold-start branch).
- [ ] Clicking Skip on a question increments `last_asked_date` but does not populate the field.
- [ ] Voice input: clicking mic, saying a sentence, releasing mic drops text into the review box.
- [ ] Submitting a rating answer updates the info card with a yellow flash and the coverage meter increments.

**Commit after each wire-up works.**

---

## Phase 12 — Demo polish, deployment, submission

Every deliverable the workbook requires.

### Task 12.1: README

**Files:** Create: `question-engine/README.md`. Must cover:
- One-paragraph problem statement (from the workbook).
- One-paragraph solution summary.
- Architecture diagram (ASCII or PNG — one of each is fine).
- `Setup` (env, `pip install -e .`, `cp .env.example .env`).
- `Build` (`python run.py build` — note it takes a few minutes first time).
- `Run` (`streamlit run app/streamlit_app.py`).
- `Data assumptions & limitations` (description versioning, PII, cold-start, adversarial reviews).
- License: MIT (required by GitHub Classroom — do not change).

### Task 12.2: Hosting

**Target:** Streamlit Community Cloud (free, simplest for Streamlit). Fork the GitHub Classroom repo if needed; the Classroom repo remains the source of truth. In Streamlit Cloud: link to the GitHub repo, select `app/streamlit_app.py`, set `OPENAI_API_KEY` via the Secrets panel, deploy.

Checklist before clicking "Deploy":
- [ ] `.env` is in `.gitignore` and not committed.
- [ ] `grep -r "sk-" question-engine` returns nothing except `.env.example`.
- [ ] `state.sqlite` is either committed as a prebuilt artifact (OK — it's derived public data) OR the app builds on first boot (slower; avoid for judge demos).
- [ ] MIT LICENSE file present at the Classroom repo root.
- [ ] Public URL loads; picking a property responds within ~3 seconds.

### Task 12.3: Demo video

**Files:** Create: `docs/video_script.md` (script only; recording is manual).

3:30–4:00 target length. Record in real time (speed-up is disqualifying).

**Script beats:**
1. **0:00–0:30** — Problem. Show a slide: "5 of Expedia's 15 rating sub-categories are 100% empty across 6,000 reviews. Static review forms don't ask."
2. **0:30–1:00** — Architecture diagram, 20 seconds.
3. **1:00–1:40** — Demo A: a data-rich property (Bell Gardens). Write: "Loved the pool, kids had fun." Show the follow-up targets an empty rating (e.g., `checkin`). Answer it. Info card flashes.
4. **1:40–2:20** — Demo B: cold-start property (Freudenstadt, 8 reviews). Empty review box, hit submit. Show the question targets a structural description gap (e.g., spa info). Answer with voice. Info card flashes.
5. **2:20–3:00** — Coverage meter before/after across the two demos.
6. **3:00–3:30** — Feasibility + limitations (version history, PII).
7. **3:30–3:50** — Why this stands out: deterministic scoring, multilingual, voice, single-file state.

### Task 12.4: Pitch deck

**Files:** Create: `docs/pitch_deck_outline.md`. 10 slides max:

1. Title + one-line pitch
2. Problem (Expedia's own quote + the 100%-null stat)
3. Definitions (what "missing" and "stale" mean)
4. Architecture diagram
5. The ranker (the math — with weights visible)
6. Demo A (screenshots)
7. Demo B (cold-start screenshots)
8. Coverage metric: before/after chart
9. Feasibility / scalability / limitations
10. What's next + team

### Task 12.5: SurveyMonkey Apply submission checklist

The day before the deadline (**Thursday 2026-04-16 09:00 ET**):

- [ ] Public prototype URL loads in an incognito window.
- [ ] GitHub Classroom repo is pushed with LICENSE intact and no API key.
- [ ] Demo video uploaded (mp4, ≤1 GB, 3–5 min, real-time).
- [ ] Pitch deck uploaded (PDF).
- [ ] Supporting materials folder has the data citation (the provided Expedia CSVs are the only data source).
- [ ] Team leader submits all materials (not a team member).

---

## Phase 13 — Stretch goals (only if green by Wednesday evening)

Ordered by marginal demo impact per hour of build time:

1. **Reasoning chip on every question**: LLM-generated one-liner, cached, shown under the question — directly addresses "why this question, why now."
2. **Property comparison view**: side-by-side of two properties' coverage — highlights the scalability story.
3. **Per-reviewer session memory**: don't re-ask a field this session that the user already skipped.
4. **"What-if" slider** for weights in the UI — lets judges see the ranker react live. Strong "technical implementation" point.
5. **Keyword-cluster radar chart** for a property — visual replacement for the list view of sub-ratings.

Do not start stretch goals until every box in Phase 12 is checked.

---

## Cross-cutting testing strategy

- **Unit tests** (Phases 2, 3, 4, 7, 8) cover all pure logic: models, ingestion, taxonomy, EMAs, scoring, ranker. These must stay green after every commit.
- **Integration test** (Phase 10) wires the flow end-to-end with a mocked LLM.
- **Manual verification** (Phase 11) is the acceptance test for the UI — checkbox list in Task 11.3.
- **No test** exercises real OpenAI calls; every LLM-facing test mocks `LlmClient`.
- Before the submission deadline: run `pytest` and confirm green. Paste the output into `docs/last_test_run.txt` for the video recording.

## Cross-cutting safety

- Pre-commit hook at `question-engine/.git/hooks/pre-commit` greps for `sk-` in staged files and fails if found. (Optional but cheap insurance.)
- `.gitignore` entries for `.env`, `data/state.sqlite`, `data/cache/` verified at end of Phase 1.
- Streamlit Cloud reads the API key from Secrets, never from a file in the repo.
- README's "Data assumptions & limitations" section names every shortcut (no description version history, no adversarial review detection, small property sample).
