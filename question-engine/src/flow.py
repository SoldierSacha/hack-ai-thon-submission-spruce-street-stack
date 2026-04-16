from __future__ import annotations
import math
from contextlib import contextmanager
from datetime import date
from pathlib import Path
import time
import numpy as np
from src.db import Repo
from src.llm import LlmClient
from src.models import (
    Answer, ContradictionAlert, FieldState, PipelineStep, Property, Question,
    Review, RatingBreakdown, TaxonomyTopic, ScoredField, EnrichmentMeta,
    SubmitResult, TagInfo, SUB_RATING_KEYS,
)
from src.enrich import detect_language, translate_to_english, tag_review
from src.ranker import rank_fields, pick_k
from src.renderer import render_question
from src.parser import parse_answer
from src.scoring import find_contradictions
from src.signals import (
    build_rating_field_states,
    build_schema_field_states,
    build_topic_field_states,
)


@contextmanager
def _timed_step(step_id: str, label: str, steps: list[PipelineStep]):
    step = PipelineStep(step_id=step_id, label=label, status="running")
    steps.append(step)
    t0 = time.perf_counter()
    try:
        yield step
        step.status = "done"
    except Exception:
        step.status = "done"
        raise
    finally:
        step.duration_ms = (time.perf_counter() - t0) * 1000


class AskFlow:
    def __init__(
        self,
        repo: Repo,
        llm: LlmClient,
        taxonomy: list[TaxonomyTopic],
        topic_embeddings: dict[str, np.ndarray],
        field_cluster: dict[str, str],
        weights_path: str | Path = "config/weights.yaml",
        cross_ref_path: str | Path = "config/cross_ref.yaml",
    ):
        self.repo = repo
        self.llm = llm
        self.taxonomy = taxonomy
        self.topic_by_id = {t.topic_id: t for t in taxonomy}
        self.topic_embeddings = topic_embeddings
        self.field_cluster = field_cluster
        self.weights_path = str(weights_path)
        self.cross_ref_path = str(cross_ref_path)
        # Track pending questions per property so submit_answer can find the review_id.
        self._pending_review_id: dict[str, str] = {}

    def submit_review(
        self, property_id: str, review_text: str, today: date,
        rating: RatingBreakdown | None = None,
        review_title: str | None = None,
    ) -> SubmitResult:
        steps: list[PipelineStep] = []

        # 1. Build a synthetic review_id
        review_id = f"{property_id}:live:{int(time.time() * 1000)}"

        # 2. Detect language
        with _timed_step("detect_lang", "Detect language", steps) as step:
            lang = detect_language(review_text) if review_text else "unknown"
            step.summary = f"Detected: {lang.upper()}" if lang != "unknown" else "No text provided"
            step.detail = {"lang": lang, "text_length": len(review_text or "")}

        # 3. Translate to English
        with _timed_step("translate", "Translate to English", steps) as step:
            if lang not in ("en", "unknown") and review_text:
                text_en = translate_to_english(review_text, lang, self.llm)
                step.summary = f"Translated from {lang.upper()}"
                step.detail = {"source_lang": lang,
                               "original_snippet": (review_text or "")[:200],
                               "translated_snippet": (text_en or "")[:200]}
            else:
                text_en = review_text if review_text else None
                step.summary = "Native English" if lang == "en" else "Skipped (no text)"
                step.status = "skipped" if not review_text else "done"

        # 4. Compute embedding
        with _timed_step("embed", "Compute embedding", steps) as step:
            review_embedding = None
            embedding_dim = 0
            if text_en and text_en.strip():
                vec = self.llm.embed(text_en)
                review_embedding = np.array(vec, dtype=np.float32)
                embedding_dim = len(vec)
                step.summary = f"{embedding_dim}-dim vector"
            else:
                step.summary = "Skipped (no text)"
                step.status = "skipped"

        # 5. Tag against taxonomy
        with _timed_step("tag_topics", "Tag review topics", steps) as step:
            if text_en:
                tags = tag_review(text_en, self.taxonomy, self.llm)
            else:
                tags = {
                    t.topic_id: {"mentioned": False, "sentiment": None, "assertion": None}
                    for t in self.taxonomy
                }
            mentioned_ids = [tid for tid, v in tags.items() if v.get("mentioned")]
            assertions_found = {tid: v["assertion"] for tid, v in tags.items()
                                if v.get("assertion")}
            step.summary = f"{len(mentioned_ids)} topics, {len(assertions_found)} assertions"
            step.detail = {"mentioned_topics": mentioned_ids, "assertions": assertions_found}

        # 6. Persist the review + embedding + tags
        with _timed_step("persist_review", "Save to database", steps) as step:
            rev = Review(
                review_id=review_id,
                eg_property_id=property_id,
                acquisition_date=today,
                rating=rating or RatingBreakdown(),
                review_title=review_title or None,
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
            step.summary = "Review + tags persisted"

        # 7. Re-aggregate field_state for this property
        with _timed_step("reaggregate", "Update field states", steps) as step:
            self._reaggregate_property(property_id)
            step.summary = "Field states refreshed"

        # 8. Rank all fields with cross-reference scoring
        with _timed_step("rank_fields", "Rank knowledge gaps", steps) as step:
            states = self.repo.list_field_states_for(property_id)
            total_reviews = len(self.repo.list_reviews_for(property_id))
            ranked = rank_fields(
                property_id=property_id,
                field_states=states,
                today=today,
                field_cluster=self.field_cluster,
                weights_path=self.weights_path,
                cross_ref_path=self.cross_ref_path,
                total_reviews=total_reviews,
            )
            step.summary = f"{len(ranked)} fields scored"
            step.detail = {"total_fields": len(ranked), "top_3": [
                {"field_id": sf.field_state.field_id, "score": round(sf.composite, 3)}
                for sf in ranked[:3]
            ]}

        # 9. Build exclusion set + pick questions
        with _timed_step("select_questions", "Filter & select questions", steps) as step:
            excluded_fids: set[str] = set()

            if rating:
                for key in SUB_RATING_KEYS:
                    if getattr(rating, key, None) is not None:
                        excluded_fids.add(f"rating:{key}")

            for tid, tag_info in tags.items():
                if tag_info.get("mentioned") and tag_info.get("sentiment") is not None:
                    excluded_fids.add(f"topic:{tid}")

            for sf in ranked:
                fid = sf.field_state.field_id
                if fid.startswith("schema:") and sf.cross_ref < 0.01:
                    excluded_fids.add(fid)

            pick_candidates = [sf for sf in ranked if sf.field_state.field_id not in excluded_fids]
            picks = pick_k(pick_candidates, field_cluster=self.field_cluster)
            step.summary = f"{len(excluded_fids)} excluded, {len(picks)} picked"
            step.detail = {"excluded_count": len(excluded_fids),
                           "picked_fields": [sf.field_state.field_id for sf in picks]}

        # 10. Render each picked field into a Question
        with _timed_step("render_questions", "Generate follow-up questions", steps) as step:
            prop = self.repo.get_property(property_id)
            questions: list[Question] = []
            for sf in picks:
                fs = sf.field_state
                topic = None
                if fs.field_id.startswith("topic:"):
                    topic = self.topic_by_id.get(fs.field_id.split(":", 1)[1])

                assertions = self.repo.get_recent_assertions(property_id, fs.field_id)

                cross_ref_context = None
                is_listing_gap = False
                if sf.cross_ref > 0.01:
                    if fs.field_id.startswith("topic:"):
                        cross_ref_context = (
                            "The property listing includes related amenities "
                            "but no guest has confirmed them."
                        )
                    elif fs.field_id.startswith("schema:"):
                        is_listing_gap = True
                        cross_ref_context = (
                            "Guests have mentioned this in reviews "
                            "but the listing field is empty."
                        )

                q = render_question(
                    field_state=fs, property_=prop, topic=topic, llm=self.llm,
                    assertions=assertions,
                    cross_ref_context=cross_ref_context,
                    is_listing_gap=is_listing_gap,
                )
                questions.append(q)
            step.summary = f"{len(questions)} questions generated"

        # 11. Detect contradictions (listing says X, reviews say X is bad/gone)
        with _timed_step("contradictions", "Detect listing contradictions", steps) as step:
            import yaml as _yaml
            _xref = _yaml.safe_load(Path(self.cross_ref_path).read_text())
            _s2t = _xref.get("schema_to_topics", {})
            _peer = {fs.field_id: fs for fs in states if fs.eg_property_id == property_id}
            raw_contradictions = find_contradictions(_peer, _s2t)
            contradiction_alerts: list[ContradictionAlert] = []
            for schema_key, topic_id in raw_contradictions:
                tfs = _peer.get(f"topic:{topic_id}")
                topic_obj = self.topic_by_id.get(topic_id)
                assertions = self.repo.get_recent_assertions(
                    property_id, f"topic:{topic_id}", sentiment_filter=-1,
                )
                alert = ContradictionAlert(
                    schema_field=schema_key,
                    topic_id=topic_id,
                    topic_label=topic_obj.label if topic_obj else topic_id,
                    sentiment=tfs.short_ema if tfs else 0.0,
                    mention_count=tfs.mention_count if tfs else 0,
                    recent_assertions=assertions[:5],
                    summary=(
                        f"The listing includes \"{schema_key.replace('_', ' ')}\" "
                        f"but {tfs.mention_count if tfs else 0} reviews rate "
                        f"\"{topic_obj.label if topic_obj else topic_id}\" negatively."
                    ),
                )
                contradiction_alerts.append(alert)
            step.summary = f"{len(contradiction_alerts)} contradictions found"
            step.detail = {"contradictions": [
                {"schema": a.schema_field, "topic": a.topic_id, "sentiment": round(a.sentiment, 2)}
                for a in contradiction_alerts
            ]}

        # Remember which review this set of Questions belongs to.
        self._pending_review_id[property_id] = review_id

        # Build enrichment metadata
        topics_tagged = sum(1 for v in tags.values() if v.get("mentioned"))
        enrichment = EnrichmentMeta(
            lang=lang,
            translated=(lang not in ("en", "unknown") and text_en is not None),
            embedding_dim=embedding_dim,
            topics_tagged=topics_tagged,
            topics_total=len(self.taxonomy),
            original_text=(review_text or "")[:300],
            translated_text=(text_en or "")[:300] if lang not in ("en", "unknown") else "",
            detected_topics=mentioned_ids,
            assertions_found=assertions_found,
        )

        total_ms = sum(s.duration_ms for s in steps)
        return SubmitResult(
            questions=questions,
            scored_fields=ranked,
            enrichment=enrichment,
            total_fields=len(ranked),
            tags={tid: TagInfo(**v) for tid, v in tags.items()},
            pipeline_steps=steps,
            total_pipeline_ms=total_ms,
            contradictions=contradiction_alerts,
        )

    def submit_answer(
        self,
        property_id: str,
        question: Question,
        answer_text: str | None,
        today: date,
    ) -> Answer:
        review_id = self._pending_review_id.get(property_id)
        if review_id is None:
            # Fallback: cache was evicted between submit_review and submit_answer
            # (common in Streamlit when the file watcher or memory pressure clears
            # @st.cache_resource). Use the most recent live review for this property,
            # or synthesize a review_id if none exists.
            reviews = self.repo.list_reviews_for(property_id)
            live = [r for r in reviews if r.source == "live"]
            review_id = live[-1].review_id if live else f"{property_id}:live:{int(time.time() * 1000)}"

        # Parse the answer
        answer = parse_answer(question, answer_text, self.llm)
        # Persist
        self.repo.record_answer(review_id, answer)

        # If scored, update the field_state immediately for UI feedback
        if answer.status == "scored":
            fs = self.repo.get_field_state(property_id, question.field_id)
            if fs is not None:
                fs.value_known = True
                fs.mention_count = (fs.mention_count or 0) + 1
                fs.last_confirmed_date = today
                # Fold a numeric answer into the EMA so the UI shows a
                # real score instead of "?".
                if isinstance(answer.parsed_value, (int, float)):
                    val = float(answer.parsed_value)
                    for attr, hl in (("short_ema", 5), ("long_ema", 30)):
                        prev = getattr(fs, attr)
                        if prev is not None:
                            alpha = 1 - math.exp(-math.log(2) / hl)
                            setattr(fs, attr, alpha * val + (1 - alpha) * prev)
                        else:
                            setattr(fs, attr, val)
                self.repo.upsert_field_state(fs)

        return answer

    def _reaggregate_property(self, property_id: str) -> None:
        """
        Re-run the three field-state builders for ONE property's worth of data.
        Much cheaper than global re-aggregation.

        Answer-confirmed knowledge is preserved: if a previous follow-up answer
        marked a field as known but the review-only rebuild would say unknown,
        we keep the answer's value_known, EMA, and confirmed date.
        """
        # Snapshot existing states so answer-based updates survive.
        old_states = {
            fs.field_id: fs
            for fs in self.repo.list_field_states_for(property_id)
        }

        properties = [self.repo.get_property(property_id)]
        reviews = self.repo.list_reviews_for(property_id)
        # review_tags filtered to this property
        all_tags = self.repo.list_review_tags_for_all()
        tags = {rid: t for rid, t in all_tags.items() if rid.startswith(f"{property_id}:")}
        rating_states = build_rating_field_states(reviews)
        schema_states = build_schema_field_states(properties, reviews=reviews)
        topic_states = build_topic_field_states(reviews, tags, self.taxonomy)

        for fs in rating_states + schema_states + topic_states:
            old = old_states.get(fs.field_id)
            if old is not None:
                # If an answer previously confirmed this field but the
                # review-only rebuild can't see it, preserve the answer.
                if old.value_known and not fs.value_known:
                    fs.value_known = True
                    fs.short_ema = old.short_ema if fs.short_ema is None else fs.short_ema
                    fs.long_ema = old.long_ema if fs.long_ema is None else fs.long_ema
                # Keep the higher counts / most-recent date (reviews + answers).
                fs.mention_count = max(fs.mention_count, old.mention_count)
                if old.last_confirmed_date and (
                    fs.last_confirmed_date is None
                    or old.last_confirmed_date > fs.last_confirmed_date
                ):
                    fs.last_confirmed_date = old.last_confirmed_date
            self.repo.upsert_field_state(fs)
