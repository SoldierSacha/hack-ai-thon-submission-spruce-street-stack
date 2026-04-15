# Expedia Hackathon — 2026 Wharton Hack-AI-thon

This is a submission for the **2026 Wharton Hack-AI-thon**, presented by Expedia Group. The challenge theme is **"Ask What Matters: Adaptive AI for Smarter Travel Reviews"**.

## The challenge

Build a prototype that asks a traveler **1–2 smart follow-up questions** while they leave a property review (via text or voice). The goal: collect missing or outdated information about a property in a low-friction way.

Guiding question: *"What information is unknown or outdated about this property, and what's the easiest way to learn it?"*

A strong solution will:
- Identify what information is missing or stale for a given property
- Ask 1–2 targeted follow-up questions that fill those gaps
- Collect answers from travelers (voice or text) as part of the review flow
- Show how those answers could improve the property's information

### Key deadlines
- **Thu, Apr 16, 2026, 9:00 AM ET** — Submission deadline (SurveyMonkey Apply)
- **Thu, Apr 16, 2026, 5:00 PM ET** — Finalists announced
- **Fri, Apr 17, 2026, 9:00 AM–3:00 PM ET** — In-person finals

### Deliverables
- Live prototype link
- GitHub Classroom repo (MIT license required, no API keys committed)
- 3–5 min real-time demo video (mp4/avi, ≤1GB — sped-up videos are disqualified)
- 8–12 slide pitch deck (for a 10-minute finals presentation + 5 min Q&A)
- Optional supporting materials

### Judging criteria
Innovation & Creativity · Technical Implementation · UX & Design · Opportunity & Impact · Feasibility & Scalability · Presentation

---

## Repository layout

```
Expedia Hackathon/
├── CLAUDE.md                   # this file
├── hackathon resources/        # provided materials — DO NOT modify
│   ├── 2026 Hack-AI-thon Workbook.pdf
│   ├── 26-Hackaithon-Kickoff.pdf
│   ├── DICTIONARY.md
│   ├── Description_PROC.csv
│   └── Reviews_PROC.csv
└── question-engine/            # our code goes here
    ├── pyproject.toml
    ├── run.py
    └── src/
        └── question_engine.py
```

## `hackathon resources/` — provided inputs

Read-only reference material from the competition organizers. Treat as the source of truth for the problem definition and data schema.

- **`2026 Hack-AI-thon Workbook.pdf`** — Full competition brief: challenge, submission requirements, timeline, judging criteria, data dictionary.
- **`26-Hackaithon-Kickoff.pdf`** — Slides from the Apr 13 kickoff webinar (business context, constraints, success metrics).
- **`DICTIONARY.md`** — Markdown version of the data dictionary (same content as the workbook's final page).
- **`Description_PROC.csv`** — One row per property. ~12 properties. Columns include:
  - `eg_property_id` (primary key), `guestrating_avg_expedia`, `city`, `province`, `country`, `star_rating`
  - `area_description`, `property_description` (narrative text; may contain `|MASK|` tokens where the property name was redacted)
  - `popular_amenities_list`, `property_amenity_*` (per-subcategory amenity lists, JSON-encoded strings)
  - `check_in_start_time`, `check_in_end_time`, `check_out_time`, `check_out_policy`
  - `pet_policy`, `children_and_extra_bed_policy`, `check_in_instructions`, `know_before_you_go`
- **`Reviews_PROC.csv`** — One row per review, joined to properties via `eg_property_id`. ~7,200 rows. Columns:
  - `eg_property_id`, `acquisition_date` (M/D/YY format), `lob` (line of business, e.g. `HOTEL`)
  - `rating` — JSON string with sub-category scores (1–5, with **0 meaning NULL/no rating**). Sub-categories: `overall`, `roomcleanliness`, `service`, `roomcomfort`, `hotelcondition`, `roomquality`, `convenienceoflocation`, `neighborhoodsatisfaction`, `valueformoney`, `roomamenitiesscore`, `communication`, `ecofriendliness`, `checkin`, `onlinelisting`, `location`
  - `review_title`, `review_text` (multilingual — English, German, etc.)

### Data notes for the question engine
- Signals for "what's missing/stale": sub-category ratings that are mostly `0` across reviews, amenity fields that are empty/sparse in `Description_PROC`, topics absent from `review_text` aggregates, and old `acquisition_date` values relative to `know_before_you_go` or renovation mentions.
- `Description_PROC` has a tiny number of properties — small enough to eyeball. `Reviews_PROC` is the volume side and drives the gap analysis.
- Reviews are in multiple languages; plan for that (either translate before analysis or use a multilingual model).

## `question-engine/` — our code

Python ≥3.11 project managed via `pyproject.toml` (currently empty deps list). This is where all prototype code lives.

- `run.py` — entrypoint; calls `QuestionEngine.main()` via `asyncio.run`.
- `src/question_engine.py` — main class (currently a skeleton with an async `main` that does nothing).
- `.venv/` — local virtual environment (don't commit).

### Adding dependencies
Edit `pyproject.toml`'s `dependencies` list. Likely candidates for this project: `openai` (organizers provide an API key, keep it OUT of git), `pandas` for CSV handling, and whatever is chosen for the UI/voice layer.

### Running
```
cd question-engine
python run.py
```

## Working agreements / constraints

- **Never commit the OpenAI API key.** If pushed publicly (including GitHub Classroom), OpenAI auto-revokes it and a new one must be issued.
- **Keep the MIT license** on the GitHub Classroom repo.
- **Cite any external data** used beyond the provided CSVs — it must be publicly available.
- All code must be developed within the hackathon window; pre-existing work is disqualifying.
