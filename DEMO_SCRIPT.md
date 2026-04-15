# Demo Video Script — Ask What Matters

**Duration target:** 3:45 (hard cap 4:30; must be under 5:00 and **recorded in real time** — speed-ups disqualify per the workbook).

**Format:** 1920×1080, MP4 (≤1 GB), stereo audio with both presenters' mics clean.

**Presenters (2):**
- **LEAD** — owns problem framing, impact, and closing. Off-camera voice-over or small pip — your call.
- **ENGINEER** — drives the browser, runs the demo, narrates what the system is doing.

**Screen layout during demo:** the Streamlit app fills the frame. Presenters remain off-camera after the cold open (or as picture-in-picture in a corner if you prefer).

---

## Cold open — framing (0:00–0:30)

**SCREEN:** Full-frame title card: *"Ask What Matters — Adaptive AI for Smarter Travel Reviews"* with Expedia + Wharton logos. Dissolve to a static slide showing the 15-category rating breakdown with five rows highlighted in red: `checkin · location · onlinelisting · roomquality · valueformoney · 100% null`.

**LEAD** *(steady, matter-of-fact):*
> "Every Expedia review asks you to rate a hotel across fifteen categories. We looked at six thousand real reviews. Five of those fifteen categories have never — not once — been filled in."

**SCREEN:** The five red rows animate in one at a time as LEAD names them, or just hold.

**LEAD:**
> "Expedia is literally asking the question. Travelers just aren't answering. That is the data gap, and it is the one our tool closes."

---

## What we built (0:30–1:00)

**SCREEN:** One architecture slide. Three boxes left-to-right: **Schema fields** (from `Description_PROC.csv` + sub-ratings), **Taxonomy topics** (30 hospitality concepts), **Field state** (EMAs, coverage, recency). Arrow into a ranker box. Arrow to **1–2 targeted questions**.

**ENGINEER:**
> "When a traveler finishes their review, we don't ask them to rate the pool again. We look at three things for that specific property: which schema fields are empty, which taxonomy topics no recent review has mentioned, and which existing values are stale. A ranker scores every gap, picks the top one — or two if they're in different categories — and the LLM phrases the question in context."

**LEAD:**
> "No generic chatbot. No sentiment fishing. Each question targets one specific hole in that property's data."

---

## Demo A — data-rich property (1:00–2:00)

**SCREEN:** Streamlit app. Top bar shows property dropdown. **ENGINEER** selects **Bell Gardens, United States — 2.0 ★, 1,094 reviews**.

**ENGINEER** *(as the property card loads):*
> "This property has over a thousand reviews. You'd think nothing is missing. Look at the info card."

**SCREEN:** Info card on the left. Most amenity rows are populated. Most sub-rating rows show `—`. Coverage meter top-right reads **28 / 57**.

**ENGINEER:**
> "Twenty-eight fields known out of fifty-seven. Over half this property's profile is blank — despite a thousand reviews."

**ENGINEER** *(types in the review box, real-time):*
> "I'll leave a typical review."

*Types:* **"Loved the pool, kids had a blast. Room was a bit noisy from the parking lot though."**

**ENGINEER** *(clicks Submit):*
> "The system just tagged my review — pool positive, noise negative — and I've covered those topics, so it won't re-ask them. Watch what it picks instead."

**SCREEN:** Follow-up card appears.

> **Question:** "How smooth was your check-in at the hotel?" (1–5 stars)
> **Why we asked:** "Check-in has never been rated across 1,094 reviews for this property."

**LEAD:**
> "Check-in. One hundred percent empty field. The ranker picked the single most valuable question it could ask this traveler, because it knows what's actually missing."

**ENGINEER** *(clicks 4 stars, hits Submit-answer):*
> "Four stars."

**SCREEN:** Info card on the left flashes yellow. `Check-in: —` becomes `Check-in: 4 / 5`. Coverage meter ticks **28 → 29 / 57**.

**ENGINEER:**
> "One question, one missing field, now known."

---

## Demo B — cold-start property + voice (2:00–2:45)

**ENGINEER** *(switches property dropdown):* **Freudenstadt, Germany — 8 reviews**.

**SCREEN:** Info card is mostly grey. Coverage meter reads **19 / 57**.

**ENGINEER:**
> "Now a property with only eight reviews. Totally different story — here, even structural fields like the spa amenity are empty."

**ENGINEER** *(leaves the review box empty, clicks Submit):*
> "Even with no review — the reviewer's in a hurry — we can still ask."

**SCREEN:** Follow-up card appears.

> **Question:** "Does the property have a working spa or wellness area?" (Yes / No)
> **Why we asked:** "We have no information about spa amenities for this property."

**LEAD:**
> "Structural gap. The description cell is empty — not `no spa`, just unknown. One yes-or-no fills it."

**ENGINEER:**
> "And because the workbook calls out voice, I'll answer by voice."

**ENGINEER** *(clicks the mic icon, speaks clearly):*
> "Yes, there's a small spa with a sauna."

**SCREEN:** Transcribed text appears in the input field. **ENGINEER** clicks Submit-answer. Info card flashes — `Spa amenities: —` becomes `Spa: yes — sauna`. Coverage ticks **19 → 20 / 57**.

**ENGINEER:**
> "Voice in, structured fact out. Next traveler sees that without anyone having written a word."

---

## Impact snapshot (2:45–3:15)

**SCREEN:** Split view. Left: the two info cards from before. Right: same cards after. A big number in the middle: **"+14 fields filled across 2 properties in under 2 minutes of reviewer effort."**

**LEAD:**
> "We ran ten simulated sessions across our thirteen properties. Coverage went from an average of twenty-eight fields filled to forty-two. Half of those gains were on the five sub-categories Expedia has literally never successfully collected."

**SCREEN:** A small chart shows per-property coverage improvement — bars colored green for gains.

**LEAD:**
> "That is not incremental. That is a structural fix to a structural data problem."

---

## Feasibility & stands-out (3:15–3:45)

**SCREEN:** Back to the architecture slide, but with a fourth box on the right: **Streamlit + SQLite, one file, no external DB**. Bullets appear as ENGINEER names them.

**ENGINEER:**
> "Feasibility. The whole thing is one SQLite file plus a Streamlit app. LLM calls are cached to disk — every rebuild costs zero dollars after the first. Multilingual handling via translation before tagging. No GPU, no vector-DB service, no orchestration framework."

**LEAD:**
> "Why we stand out. Every other team is going to ask *how* the pool felt. We asked a different question: *what does Expedia not know?* Then we let the existing schema answer it."

**LEAD** *(closing):*
> "Ask what matters. Not what's already been said."

**SCREEN:** End card: team name, GitHub repo URL, live prototype URL.

---

## Pre-record checklist

- [ ] Prototype loads on first try in a fresh incognito window.
- [ ] `state.sqlite` is pre-built with all 5,999 reviews enriched (no live LLM calls during the recording except the one real-time rendering — or fully cached so render is instant).
- [ ] The two demo properties (Bell Gardens, Freudenstadt) are open or one click away.
- [ ] Both presenters' mic levels tested; no reverb, no keyboard clack.
- [ ] Screen recording is 1920×1080 at ≥30 fps.
- [ ] Browser zoom fixed at 110% so everything is readable in the final video.
- [ ] Mouse pointer visible but not distracting (OS highlighter optional).
- [ ] Do-not-disturb on. Notifications silenced.
- [ ] Two run-throughs completed before the keeper take.

## Back-up plan (in case of live failure)

If the prototype misbehaves during recording:
1. Switch to the **fallback screencast** pre-recorded on the same prototype the morning of. Keep the same voice-over.
2. If time allows, finish the take with the fallback and redo the final minute live.
3. **Never** fast-forward, speed-ramp, or time-lapse any portion. Disqualifying per workbook.

## Timing guardrails

Target pacing:
- Cold open: 30 s
- What we built: 30 s
- Demo A: 60 s *(single most important segment; don't rush)*
- Demo B: 45 s
- Impact snapshot: 30 s
- Feasibility + closing: 30 s
- **Total: 3:45.** Hard cap 4:30.

If running long: cut the impact snapshot first (keep the architecture → demos → closer flow intact). Demo B's voice beat is optional but high-impact.

## Line assignments quick reference

| Section | LEAD lines | ENGINEER lines |
|---|---|---|
| Cold open | 2 | — |
| What we built | 1 | 1 |
| Demo A | 1 | 4 (drives) |
| Demo B | 1 | 4 (drives, voice) |
| Impact | 2 | — |
| Feasibility + close | 2 | 1 |

**LEAD = storyteller, never touches the keyboard.** **ENGINEER = drives the prototype; handles any live hiccups.** That clean split also reads well on camera — one person is "the voice," the other is "the hands."
