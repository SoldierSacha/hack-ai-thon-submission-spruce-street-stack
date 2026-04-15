[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/62ykpBfA)

# Ask What Matters — Adaptive AI for Smarter Travel Reviews

> **One-liner:** When a traveler finishes typing their review, our system looks at what Expedia already knows (and doesn't) about that specific property, picks the 1–2 highest-value gaps, and asks exactly those.

A submission for the **2026 Wharton Hack-AI-thon**, presented by Expedia Group. Challenge theme: *"Ask What Matters: Adaptive AI for Smarter Travel Reviews."*

---

## The problem

Every Expedia review asks the traveler to rate a property across **15 sub-categories**. We counted how often each one actually gets filled in across 5,999 real reviews spanning January 2024 to September 2025:

| Sub-rating | % empty (rating = 0) |
|---|---|
| `checkin` | **100.00%** |
| `location` | **100.00%** |
| `onlinelisting` | **100.00%** |
| `roomquality` | **100.00%** |
| `valueformoney` | **100.00%** |
| `communication` | 99.98% |
| `convenienceoflocation` | 99.98% |
| `neighborhoodsatisfaction` | 99.98% |
| `roomcomfort` | 86.6% |
| `ecofriendliness` | 67.8% |
| `roomamenitiesscore` | ~60% |
| `service` | 22.5% |
| `hotelcondition` | 19.9% |
| `roomcleanliness` | 18.9% |
| `overall` | 0% |

Five categories have **never, not once** in 5,999 reviews been answered. Another three are within one rounding error of never. Expedia is literally asking the question — travelers just aren't answering.

This is a structural data gap, not a content gap. Reviews themselves are short (median 58 characters) and multilingual (~52% English, the rest German, Spanish, Italian, French, and a long tail). The 13 properties in the dataset range from 8 reviews to 1,094 — a 137× imbalance that forces any solution to handle both cold-start and data-rich cases gracefully.

## How it works

The prototype maintains a **per-property state** across three layers and combines them in a single ranker that the LLM then renders into a natural-language question.

### Layer 1 — Schema fields

Every column in `Description_PROC.csv` (amenity lists, policies, check-in/out times, area descriptions) plus every sub-rating key in `Reviews_PROC.csv`'s `rating` JSON. The primary missing-info signal is simply: *is this cell empty?* The five 100%-null sub-ratings fall out here mechanically.

### Layer 2 — Taxonomy topics

A fixed **30-concept hospitality taxonomy** (see `question-engine/config/taxonomy.yaml`) that spans the things reviewers actually mention but that don't map cleanly to schema fields: `wifi_reliability`, `noise_level`, `breakfast_quality`, `staff_friendliness`, `elevator_availability`, and so on. The taxonomy is pluggable YAML — a property in a new region can be handled by swapping the concept list.

Every enriched review gets tagged against this taxonomy. The gap signal here is inverse coverage: topics that **no recent review mentioned** at a property become askable.

### Layer 3 — Aggregated signals (field state)

For each `(property × field)` pair we track:

- `value_known` — do we have a value at all?
- `last_confirmed_date` — when was this most recently mentioned or updated?
- `short_EMA` and `long_EMA` over sentiment on a 1–10 scale — the gap between them flags drift (e.g. a renovation changed reality).
- `mention_count` — how many reviews have touched this field.

### The ranker

For every field, three scalars:

```
missing_score  ∈ {0, 1}      — 1 if value_known is false
stale_score    ∈ [0, 1]      — age / 180 days, +0.5 if |short_EMA − long_EMA| > 0.5 and mention_count ≥ 5
coverage_gap   = 1 / (1 + mention_count)
```

Plus a redundancy penalty: cosine similarity between the current review's embedding and the topic's embedding (only applied to `topic:*` fields — schema fields can't be "redundant" to a review body).

Combined:

```
score = 0.55 * missing_score
      + 0.25 * stale_score
      + 0.15 * coverage_gap
      − 0.35 * redundancy
      + ε                    # tiny deterministic tiebreak
```

Picks top-1, or top-2 if **both** exceed `min_score = 0.4` **and** sit in different clusters (no asking two questions about the same cluster on one review).

### LLM rendering and parsing

The picked field ID + its current known value + the property's `(city, country, star_rating)` + either the taxonomy topic's question hint or a static hint for schema fields flow into a small rendering prompt. Output: a natural-language question plus a one-line "why we're asking" for the UI.

Answers come back through a typed parser:

- `rating_1_5` — regex shortcut (digits or word-to-digit), LLM JSON fallback.
- `yes_no` — regex shortcut, LLM JSON fallback.
- `short_text` — LLM JSON extraction with abstain detection (the LLM returns `{"abstain": true}` if the answer doesn't actually contain the fact asked for).

### Data flow

```
   Description_PROC.csv        Reviews_PROC.csv
          │                           │
          ▼                           ▼
  ┌────────────┐             ┌────────────────┐
  │ properties │             │  raw reviews   │
  └─────┬──────┘             └────────┬───────┘
        │                             │
        │                             ▼
        │                   ┌───────────────────┐
        │                   │ enrich: language, │
        │                   │ translate, tag    │
        │                   │ against taxonomy  │
        │                   └─────────┬─────────┘
        │                             │
        ▼                             ▼
  ┌────────────────────────────────────────────┐
  │          SQLite: state.sqlite              │
  │  properties · reviews · field_state        │
  └────────────────────┬───────────────────────┘
                       │
  user types review    ▼
  ─────────────▶ ┌──────────────────────┐
                 │   live enrich +      │
                 │   update short-EMA   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ rank all fields for  │
                 │ this property, pick  │
                 │ top 1-2 gaps         │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ LLM renders question │
                 │ + reason string      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ user answers (text   │
                 │ or voice); parser    │
                 │ returns typed value  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ write back: field_   │
                 │ state updated, info  │
                 │ card reflects gain   │
                 └──────────────────────┘
```

## Repo layout

```
Expedia Hackathon/
├── README.md                    # this file
├── CLAUDE.md                    # internal dev notes
├── PLAN.md                      # implementation plan (checked in, 25 completed phases)
├── DEMO_SCRIPT.md               # 3:45 two-presenter demo script
├── docs/
│   └── pitch_deck_outline.md    # 10-slide deck + judge Q&A prep
├── hackathon resources/         # NOT committed — dataset is Expedia's
│   ├── Description_PROC.csv
│   ├── Reviews_PROC.csv
│   └── DICTIONARY.md
└── question-engine/
    ├── pyproject.toml
    ├── run.py                   # CLI: `build` | `ask`
    ├── .env.example             # committed; real .env is gitignored
    ├── .streamlit/config.toml
    ├── config/
    │   ├── taxonomy.yaml        # 30 hospitality topics
    │   └── weights.yaml         # ranker weights (tunable without code change)
    ├── data/
    │   ├── state.sqlite         # NOT committed
    │   └── cache/               # NOT committed; SHA256-keyed LLM cache
    ├── src/
    │   ├── models.py            # Pydantic types
    │   ├── db.py                # SQLite repository
    │   ├── ingest.py            # CSV → Property / Review
    │   ├── taxonomy.py          # topic loader + normalizer
    │   ├── llm.py               # OpenAI wrapper + on-disk cache
    │   ├── enrich.py            # lang detect / translate / tag
    │   ├── signals.py           # EMAs + field_state builders
    │   ├── scoring.py           # missing / stale / coverage
    │   ├── ranker.py            # rank + pick_k + cluster de-dup
    │   ├── renderer.py          # LLM question rendering
    │   ├── parser.py            # LLM answer parsing
    │   ├── flow.py              # AskFlow orchestrator
    │   └── question_engine.py   # `build()` and `run_ask()` entrypoints
    ├── app/
    │   └── streamlit_app.py     # UI
    └── tests/                   # 85 tests across 10 files
```

## Quick start

Requires Python 3.11+ and an OpenAI API key (organizer-provided for the hackathon; never commit it).

```bash
cd question-engine
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
cp .env.example .env   # add your OPENAI_API_KEY
```

Place the two provided CSVs (`Description_PROC.csv`, `Reviews_PROC.csv`) in `../hackathon resources/`. They are not committed here — the dataset belongs to Expedia.

### Build the property state

```bash
# small smoke-test build — 50 reviews across all properties, ~24 seconds
.venv/bin/python run.py build --limit 50 --workers 4

# full build — all 5,999 reviews, ~30 minutes, ~$5 in OpenAI calls
.venv/bin/python run.py build --workers 4
```

`build` is idempotent. The LLM cache (see below) keys every call on `(system, user, model, temperature)`, so a second pass over the same reviews is free.

### Launch the UI

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Opens on `http://localhost:8501`. Pick a property, paste or dictate a review, click submit, get the follow-up question, answer it, watch the info card update.

### CLI for quick verification

```bash
# list all property IDs
.venv/bin/python run.py ask list

# single-shot ask flow, no UI
.venv/bin/python run.py ask <property_id> "Loved the pool, kids had a blast."
```

## Scaling notes

- **Cost.** A full 5,999-review enrichment takes about 30 minutes and roughly \$5 against `gpt-4.1-mini` + `text-embedding-3-small`. That cost is one-time per dataset snapshot.
- **Cache.** Every LLM and embedding call is SHA256-keyed on `(system, user, model, temperature)` and serialized to `data/cache/`. Re-runs after a crash, a parameter tweak, or a test pass cost \$0. To force re-enrichment, delete the cache directory.
- **Storage.** Everything lives in `data/state.sqlite`. A single file. For production this swaps to Postgres with zero query changes — the repo layer in `src/db.py` uses parameterized SQL.
- **Concurrency.** `--workers N` parallelizes enrichment across asyncio tasks. N=4 is the sweet spot on a MacBook — more hits OpenAI rate limits with no speedup.
- **Multilingual.** Reviews are auto-detected, translated to English before tagging, and kept alongside the original text. Currently observed: English, German, Spanish, Italian, French, and "unknown" (very short phatic reviews).

## What's in the demo

See `DEMO_SCRIPT.md` for the full two-presenter script. The demo covers two properties that represent the opposite ends of the data-imbalance spectrum:

- **Bell Gardens, United States — 2.0★, 1,094 reviews (data-rich).** Info card shows 28 / 57 fields known despite the volume. A reviewer types *"Loved the pool, kids had a blast. Room was a bit noisy from the parking lot."* The ranker ignores pool and noise (now covered) and asks about `checkin` — a 100%-null sub-rating the reviewer never thought to fill in.
- **Freudenstadt, Germany — 8 reviews (cold-start).** Info card mostly grey. Reviewer leaves the review box empty. The ranker asks about a structural amenity gap — *"Does the property have a working spa or wellness area?"* — which the user answers by **voice** via `streamlit-mic-recorder`.

Both demos show the info card flashing and the coverage meter ticking up in real time.

## Tech stack

- **Python 3.11+**, `asyncio` for concurrency.
- **OpenAI** — `gpt-4.1-mini` for enrichment, rendering, and parsing; `text-embedding-3-small` for redundancy scoring. No other model providers.
- **Pydantic v2** — all models and DTOs; strict validation at the boundaries.
- **SQLite** — single file, zero ops.
- **Streamlit** — the entire UI, including mic input via `streamlit-mic-recorder`.
- **pytest** — 85 tests across ingestion, enrichment, scoring, ranking, rendering, parsing, and the full `AskFlow` orchestration.

No vector database, no orchestration framework, no microservices. Every piece of state is inspectable by opening one `.sqlite` file in DB Browser.

### Why these choices

- **`gpt-4.1-mini` over larger models.** The tasks — language detection, topic tagging against a fixed 30-item list, short-answer extraction, one-sentence rendering — are all narrow. Larger models add latency and cost without improving accuracy on any metric we measured.
- **`text-embedding-3-small` over 3-large.** Redundancy scoring is a cosine similarity threshold, not a retrieval ranking. Small embeddings are ~5× cheaper and indistinguishable at this threshold.
- **SQLite over Postgres.** One file, zero ops, fully inspectable. The repo layer is parameterized SQL — switching to Postgres for production is a connection-string change.
- **Streamlit over a React SPA.** The UI is 4 panels and a mic button. Building it in Streamlit took an afternoon; the same UI in React would have been a day of plumbing that doesn't advance the core idea.
- **Pydantic v2 over plain dicts.** Every LLM response is validated at the boundary. A malformed JSON from the model fails the Pydantic parse and triggers the retry/abstain path, instead of propagating `None` deep into the ranker.

## Inspectability

A judge's natural first question is: *"Why did it ask that?"* The system is built so that every decision is traceable to a number in the database.

Open `data/state.sqlite` and you see three tables:

- `properties` — one row per property, one column per schema field, plus a derived `value_known` map.
- `reviews` — one row per enriched review, with original text, detected language, English translation, and taxonomy tag set.
- `field_state` — one row per `(property_id, field_id)` tuple; this is the table the ranker reads.

Query examples that answer "why did it ask that":

```sql
-- The five fields most worth asking about for property X
SELECT field_id, missing_score, stale_score, coverage_gap, mention_count
FROM field_state
WHERE property_id = ?
ORDER BY (0.55*missing_score + 0.25*stale_score + 0.15*coverage_gap) DESC
LIMIT 5;

-- Every field that has never been mentioned on any review for this property
SELECT field_id FROM field_state
WHERE property_id = ? AND mention_count = 0;

-- Where has sentiment drifted recently?
SELECT field_id, short_ema, long_ema, ABS(short_ema - long_ema) AS drift
FROM field_state
WHERE property_id = ? AND mention_count >= 5
ORDER BY drift DESC LIMIT 10;
```

Every row the ranker ever saw is still sitting there — no logs to grep, no traces to decode.

## What we explicitly don't solve

Honest limitations. The workbook rewards acknowledging them.

- **No description version history.** `Description_PROC.csv` is a snapshot — we assume each property's description was valid as of the newest review date for that property. When a description is stale we can detect sentiment drift on related fields but we can't prove the description row itself is out of date.
- **No PII scrubbing beyond the `|MASK|` tokens already in the workbook data.** The LLM is instructed not to quote reviewer text in questions, but we do not run a separate NER redaction pass.
- **No adversarial-review detection beyond LLM abstain.** A traveler who types deliberate nonsense or tries to prompt-inject gets the abstain path in the parser and nothing is written to state. We do not separately classify reviews as spam.
- **No cross-property benchmarking.** We rank gaps within a property, not across properties. A question engine that said "your property scores low on breakfast vs peers" is out of scope.
- **No personalization across sessions.** The dataset has no user IDs. We can't remember "this traveler already told us X last trip."

## Data citation

The two CSVs this prototype reads — `Description_PROC.csv` and `Reviews_PROC.csv` — are provided by Expedia Group as part of the 2026 Wharton Hack-AI-thon materials. They are not redistributed in this repository. Teams running this code outside the hackathon need to supply their own property and review data matching the same schema (see `hackathon resources/DICTIONARY.md` for the field list).

No external data sources beyond the provided CSVs are used.

## License

MIT, per hackathon requirement. The repo retains the default `LICENSE` file from its GitHub Classroom template.

## Team

{{team name here}}

---

*"Ask what matters. Not what's already been said."*
