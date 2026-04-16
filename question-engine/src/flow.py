from __future__ import annotations
import math
from datetime import date
from pathlib import Path
import time
import numpy as np
from src.db import Repo
from src.llm import LlmClient
from src.models import (
    Answer, FieldState, Property, Question, Review, RatingBreakdown, TaxonomyTopic,
    ScoredField, EnrichmentMeta, SubmitResult,
)
from src.enrich import detect_language, translate_to_english, tag_review
from src.ranker import rank_fields, pick_k
from src.renderer import render_question
from src.parser import parse_answer
from src.signals import (
    build_rating_field_states,
    build_schema_field_states,
    build_topic_field_states,
)


class AskFlow:
    def __init__(
        self,
        repo: Repo,
        llm: LlmClient,
        taxonomy: list[TaxonomyTopic],
        topic_embeddings: dict[str, np.ndarray],
        field_cluster: dict[str, str],
        weights_path: str | Path = "config/weights.yaml",
    ):
        self.repo = repo
        self.llm = llm
        self.taxonomy = taxonomy
        self.topic_by_id = {t.topic_id: t for t in taxonomy}
        self.topic_embeddings = topic_embeddings
        self.field_cluster = field_cluster
        self.weights_path = str(weights_path)
        # Track pending questions per property so submit_answer can find the review_id.
        self._pending_review_id: dict[str, str] = {}

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
