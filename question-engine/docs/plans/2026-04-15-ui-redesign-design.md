# UI Redesign: Coverage Radar Dashboard

**Date**: 2026-04-15
**Goal**: Redesign the Streamlit UI to surface the question engine's technical features (gap analysis, scoring, enrichment pipeline, taxonomy, field states) directly in an integrated dashboard that impresses hackathon judges.

## Approach

**Coverage Radar Dashboard** — radar chart + color-coded field inventory + score breakdowns on question cards + enrichment summary strip. All on a single page, two-column layout.

## Layout

```
+---------------------------------------------------------------+
|  [Navy header bar]  Ask What Matters         [Property v] [R] |
+---------------------------+-----------------------------------+
| PROPERTY KNOWLEDGE PANEL  | REVIEW & QUESTION FLOW            |
|                           |                                   |
| [Property card]           | [Review input area]               |
|  City, Country            |  Text area + voice button         |
|  Star badge, Guest avg    |  [Submit review]                  |
|  Check-in/out times       |                                   |
|                           | [Enrichment summary] (post-submit)|
| [Donut: X/Y known]       |  Lang: de | Translated | 384-dim  |
|                           |  Topics: 8/28 tagged              |
| [Radar chart: 11 clusters]|                                   |
|  connectivity, room, ...  | [Question card #1]                |
|  % known per cluster      |  Question text (bold)             |
|                           |  "Why?" expander:                 |
| [Field inventory]         |    Score = 0.72 breakdown         |
|  Sub-ratings (color-coded)|    Rank #1 of 58 | Cluster: Room  |
|  Schema fields            |  [Input widget]                   |
|  Topics by cluster        |  [Submit] [Skip]                  |
|                           |                                   |
|                           | [Question card #2] ...            |
+---------------------------+-----------------------------------+
```

## Section Details

### 1. Header Bar

- Full-width navy (`#00355F`) background, white title
- Property selectbox with enriched labels: `City, Country (N* , X reviews, Y% covered)`
- Reset button (subtle, secondary style)

### 2. Property Knowledge Panel (left column)

**Property card**: city/country, star rating pill, guest average, check-in/out. Styled card with shadow.

**Coverage donut** (Plotly): single ring showing known vs total fields. Large center number.

**Radar chart** (Plotly): 11 axes (one per cluster), each showing % of fields known in that cluster. Updates live on answer submission.

**Field inventory**: three sections (sub-ratings, schema fields, topic clusters), each with:
- **Color-coded status pills**:
  - Green: "Known" (value_known=True, confirmed < 90 days ago)
  - Amber: "Stale" (confirmed > 90 days ago) or "Drifting" (|short_ema - long_ema| >= 0.3)
  - Red: "Missing" (value_known=False)
- **Mini score bars**: thin horizontal bar per field showing missing/stale/coverage segments
- Topics grouped by cluster in expanders with sentiment icons

### 3. Review & Question Flow (right column)

**Review input**: text area + mic button in a styled container.

**Enrichment summary** (appears after submit): compact 4-metric strip:
- Language detected
- Translation status
- Embedding dimensions
- Topics tagged count

**Question cards**: each shows:
- Question text (large, 18px, bold, indigo border)
- "Why this question?" expander:
  - Reason text
  - Score formula: `Score = w_m * missing + w_s * stale + w_c * coverage - w_r * redundancy`
  - Actual values filled in
  - Rank badge: "#1 of N fields"
  - Cluster label
- Type-appropriate input widget
- Submit + Skip buttons

### 4. Styling

- **Palette**: Navy `#00355F`, accent yellow `#FBCE38`, success green `#16a34a`, warning amber `#f59e0b`, danger red `#ef4444`, light bg `#F5F5F5`, cards white
- Cards: `border-radius: 12px`, `box-shadow: 0 1px 3px rgba(0,0,0,0.1)`
- Clean typography hierarchy via custom CSS

### 5. New Dependency

- `plotly>=5.18` (Streamlit has native `st.plotly_chart`)

## Data Flow Changes

**No backend changes.** All changes are in `streamlit_app.py`. The scoring/ranking data needed for the question card breakdowns requires passing intermediate results from `flow.submit_review()` through to the UI:

- `flow.submit_review()` currently returns `list[Question]`. We need to also surface:
  - The enrichment metadata (language, translation status, topic count)
  - The scored/ranked field list (so we can show rank position and score breakdown)
  - The review embedding (already computed internally)

Options:
1. Add a `SubmitResult` dataclass that bundles questions + metadata
2. Store enrichment metadata in session state via flow method that returns richer data

**Recommended**: Option 1 — add a small `SubmitResult` model and modify `submit_review()` to return it. Minimal change, clean interface.

## Files Changed

- `src/models.py` — add `SubmitResult` dataclass
- `src/flow.py` — modify `submit_review()` return type
- `app/streamlit_app.py` — full rewrite of the UI
- `pyproject.toml` — add `plotly` dependency
