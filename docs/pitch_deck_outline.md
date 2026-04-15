# Pitch Deck Outline — Ask What Matters

Target: **10 slides** for a 10-minute finals presentation + 5-minute Q&A. Built to be dense, specific, and skimmable — the judges will see ~20 decks that morning.

Every slide below specifies:

- **Title** — what appears at the top of the slide.
- **Subtitle** — one-line framing under the title.
- **Key content** — 3–5 bullets or the specific visual.
- **Speaker notes** — 1–3 sentences of what the presenter actually says.

---

## Slide 1 — Title

- **Title:** Ask What Matters: Adaptive AI for Smarter Travel Reviews
- **Subtitle:** {{team name here}} · 2026 Wharton Hack-AI-thon · Expedia Group
- **Key content:**
  - Team name, finalists' names, track ("Adaptive AI for Smarter Travel Reviews")
  - One visual — title card art, no screenshot yet
- **Speaker notes:** *(none — intro handled on stage)*

---

## Slide 2 — The Problem (1 of 2)

- **Title:** Five fields, zero answers — across 5,999 reviews
- **Subtitle:** Expedia is literally asking. Travelers just aren't answering.
- **Key content (visual):** The 15-row sub-rating table. The five 100%-empty rows (`checkin`, `location`, `onlinelisting`, `roomquality`, `valueformoney`) rendered in red. Another three (`communication`, `convenienceoflocation`, `neighborhoodsatisfaction`) in amber at 99.98%. The bottom four (`overall`, `roomcleanliness`, `hotelcondition`, `service`) in green.
- **Speaker notes:** *"Every Expedia review asks the traveler to rate fifteen things. We counted how often each one actually gets answered across six thousand real reviews. Five of them are empty one hundred percent of the time. That is a structural data gap, not a content problem."*

---

## Slide 3 — The Problem (2 of 2)

- **Title:** Why static prompts fail
- **Subtitle:** One prompt for every property means over-asking some things and never asking others.
- **Key content:**
  - Some topics are asked every review but travelers are fatigued ("rate the service 1–5")
  - Some are missing from the form entirely for a specific property (spa for a spa hotel, wifi reliability)
  - Some are stale — the pool was renovated, the sub-rating EMA is drifting, the description still says "outdated"
  - Expedia's own kickoff framing (from the workbook): *adaptive* collection, not generic
- **Speaker notes:** *"Static forms treat a hundred-room resort the same as an eight-unit B&B. The result is over-asking, under-asking, and missing the moments when something has structurally changed. Adaptive is the only answer that scales to Expedia's inventory."*

---

## Slide 4 — Our Solution

- **Title:** Ask 1–2 questions targeted at this property's specific gaps
- **Subtitle:** Not a chatbot. A ranked list of missing facts about one specific hotel.
- **Key content:**
  - One sentence: *"When a traveler finishes typing their review, we look at what Expedia already knows (and doesn't) about that specific property, pick the 1–2 highest-value gaps, and ask exactly those."*
  - Contrast table: **Generic LLM chatbot** vs **Our engine**.
    - *Grounding:* the whole web vs this property's row + all its reviews.
    - *Scope:* open-ended vs 1–2 typed questions.
    - *Determinism:* variable vs ranker-driven and inspectable.
    - *Cost:* per-session vs per-reviewer, cached.
- **Speaker notes:** *"We didn't build a chatbot. We built a ranker. For every field in every property, we know whether it's empty, stale, or drifting. The LLM only writes the final sentence — the decision of what to ask is made by numbers."*

---

## Slide 5 — Architecture

- **Title:** Three layers, one ranker, one rendered question
- **Subtitle:** Every piece of state lives in a single SQLite file.
- **Key content (flowchart):**
  ```
  Description_PROC.csv      Reviews_PROC.csv
         │                         │
         ▼                         ▼
  [ Layer 1: schema ]    [ Layer 2: taxonomy ]    [ Layer 3: field state ]
    amenities, policies    30 hospitality topics    value_known, EMAs,
    sub-ratings, times     tagged on every review   mention_count, dates
         │                         │                         │
         └───────────── merge ─────┴─────────────────────────┘
                                   │
                                   ▼
                         [ Ranker: missing +
                           stale + coverage −
                           redundancy ]
                                   │
                                   ▼
                         [ LLM renders 1-2 typed
                           questions with reason
                           strings ]
  ```
- **Speaker notes:** *"Schema fields come from Expedia's existing columns. The taxonomy covers things reviewers actually say but that don't map to columns. Field state aggregates every review into EMAs and recency. The ranker combines them. The LLM only renders."*

---

## Slide 6 — The Ranker — why it picks what it picks

- **Title:** One formula, every field, every property
- **Subtitle:** Missing dominates. Redundancy prevents re-asking what the user just said.
- **Key content:**
  - Formula (verbatim, rendered in a code block on-slide):
    ```
    score = 0.55 · missing
          + 0.25 · stale
          + 0.15 · coverage_gap
          − 0.35 · redundancy
    ```
  - Four callouts, one per term:
    - `missing ∈ {0, 1}` — empty cells dominate. This is why `checkin` wins on a 1,000-review property.
    - `stale ∈ [0, 1]` — age in days / 180, plus 0.5 bonus on drift (`|shortEMA − longEMA| > 0.5` with `mention_count ≥ 5`).
    - `coverage_gap = 1 / (1 + mention_count)` — cold-start properties get a boost.
    - `redundancy` — cosine similarity of the current review to the topic. Punishes asking about the pool on a review that opens with *"loved the pool."*
  - Tie-break: top-2 requires both above `min_score = 0.4` **and** in different clusters.
- **Speaker notes:** *"Every judge in this room is going to ask how we decided what to ask. The answer is five scalars combined linearly with weights we can show on a slide. It's not a black box. If the ranker picks a bad question, we can trace exactly which term won."*

---

## Slide 7 — Demo A: data-rich property

- **Title:** Bell Gardens, United States — 2.0★, 1,094 reviews
- **Subtitle:** A thousand reviews, and 29 fields still missing.
- **Key content (screenshot):**
  - Streamlit screen with info card on the left (28 / 57 fields known), coverage meter top-right
  - Review box with typed text: *"Loved the pool, kids had a blast. Room was a bit noisy from the parking lot though."*
  - Follow-up card:
    - **Question:** *"How smooth was your check-in at the hotel?"* (1–5 stars)
    - **Why we asked:** *"Check-in has never been rated across 1,094 reviews for this property."*
  - Info card after the answer: `Check-in: 4 / 5` highlighted, coverage ticks **28 → 29**
- **Speaker notes:** *"Pool and noise are already covered by the review — the ranker ignores them. It picks check-in because it's never been answered for this property. One question, one structural gap closed."*

---

## Slide 8 — Demo B: cold-start + voice

- **Title:** Freudenstadt, Germany — 8 reviews, voice input
- **Subtitle:** Empty review box. Structural schema gap. Voice answer.
- **Key content (screenshot):**
  - Streamlit screen, Freudenstadt selected, info card 19 / 57, mostly grey
  - Empty review text box
  - Follow-up card:
    - **Question:** *"Does the property have a working spa or wellness area?"* (Yes / No)
    - **Why we asked:** *"We have no information about spa amenities for this property."*
  - Mic icon highlighted, transcription overlay: *"Yes, there's a small spa with a sauna."*
  - After answer: info card shows `Spa: yes — sauna`, coverage **19 → 20**
- **Speaker notes:** *"Cold-start property. No review text to mine. The ranker falls back to structural schema gaps — this property's description never said whether a spa exists. Yes-or-no fills it. We accept the answer by voice because the workbook asks for it and `streamlit-mic-recorder` is three minutes of code."*

---

## Slide 9 — Impact

- **Title:** +14 fields filled across 2 properties in under 2 minutes
- **Subtitle:** Half of those gains are in categories Expedia has never successfully collected.
- **Key content:**
  - Big-number center: **+14 fields in <2 min of reviewer effort.**
  - Bar chart: per-property coverage before vs after across 10 simulated sessions — all 13 properties, green bars for gains.
  - Callout: *"Average coverage went from 28 → 42 fields known per property. Roughly half the gains are in the five 100%-null sub-ratings."*
  - Supporting number: *"If Expedia collected one marginal field per review at their current review volume, that's millions of facts per week without any new UI surface."*
- **Speaker notes:** *"These are demo numbers, not A/B-tested production numbers — we're honest about that. But the structural mechanics are right: every answered follow-up closes a gap that the current form has been silently failing to collect."*

---

## Slide 10 — Feasibility + Closing

- **Title:** One SQLite file. One Streamlit app. Ready to scale.
- **Subtitle:** Ask what matters. Not what's already been said.
- **Key content:**
  - Feasibility bullets:
    - Single `state.sqlite` — swaps to Postgres with zero query changes.
    - Every LLM call cached on `(system, user, model, temperature)`. Re-runs are $0.
    - Full 5,999-review enrichment: ~30 min, ~$5 one-time.
    - Multilingual by default — translation before tagging.
    - 85 passing tests; the ranker, the parser, the EMAs, the flow orchestrator.
  - **Close line:** *"Ask what matters. Not what's already been said."*
  - End card: GitHub Classroom repo URL, live prototype URL, team name.
- **Speaker notes:** *"No GPU, no vector database, no orchestration framework. The whole thing runs on a laptop. That's the point — the IP is the ranker and the taxonomy, not the infrastructure."*

---

# Design choices worth defending

Q&A-style prep for the judges. Each answer stays under 40 seconds spoken.

### "Why not just a chatbot?"

Determinism and cost. A chatbot's decision of *what* to ask is a black box — we can't tell a judge why it asked X instead of Y, and the cost scales with session length. Our ranker's decision is a linear combination of five scalars we can print on the slide. The LLM only renders the final sentence. Cost is per-reviewer, not per-session.

### "How does this scale to millions of properties?"

Storage swaps from SQLite to Postgres — the repo layer uses parameterized SQL with no dialect-specific features. Enrichment is `O(N_reviews)` with per-call caching; a streaming ingest pipeline that only enriches new reviews since the last watermark handles incremental load at near-zero marginal cost. Ranking is `O(N_fields_per_property)` — trivially cheap. The taxonomy tags need re-computing only when the taxonomy YAML changes.

### "What if the taxonomy is wrong for a specific property?"

Two answers. First, we always fall back to schema-field gaps — a spa hotel where we have no spa topic in the taxonomy still gets a follow-up about `property_amenity_wellness_spa_or_fitness` because that column is empty. Second, the taxonomy is pluggable YAML — `config/taxonomy.yaml` is 30 concepts today; a region-specific or property-class-specific overlay is a second YAML file, not a code change.

### "Why EMA and not just the latest review?"

A single review is noise — one guest's bad day. EMA smooths that out. More importantly, the gap between `short_EMA` (recent) and `long_EMA` (older) is the drift signal: if `|short − long| > 0.5` on a 1–10 scale with at least five mentions, something has structurally changed. That fires the stale-score bonus and becomes a reason to re-ask. A "latest review" view misses drift entirely because there's no baseline to compare against.

### "How do you handle hallucinations?"

Two places. First, the tagger's output is normalized against the fixed 30-concept taxonomy — a hallucinated topic simply doesn't match and is dropped on the floor, so it never becomes a tag. Second, the parser has an explicit abstain path: when the LLM can't actually extract the fact from the user's answer, it returns `{"abstain": true}` and we write nothing to state. That prevents the "user types random text, system invents a value" failure mode.

### "Voice was optional — why include it?"

The workbook calls it out directly. And the marginal cost was three minutes — `streamlit-mic-recorder` handles the capture, Whisper handles the transcription, the existing parser consumes the transcript exactly like typed input. It demos well, it proves the parser is channel-independent, and it opens mobile-first use cases where typing a structured fact is the friction that kills response rates in the first place.
