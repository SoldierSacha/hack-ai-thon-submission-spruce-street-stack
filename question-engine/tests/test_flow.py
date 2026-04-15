from datetime import date
from unittest.mock import MagicMock
import numpy as np
from pathlib import Path
from src.db import Repo
from src.models import Property, Review, RatingBreakdown, TaxonomyTopic, Question
from src.flow import AskFlow
from src.taxonomy import load_taxonomy
from src.ranker import build_field_cluster_map


def _setup_flow(tmp_path):
    """Prepare a flow with a mocked LLM and a tiny seeded repo."""
    repo = Repo(tmp_path / "state.sqlite")
    repo.init_schema()
    # Seed a property with no prior reviews
    repo.upsert_property(Property(eg_property_id="p1", city="Pompei", country="Italy",
                                  star_rating=3.5, amenities={"spa": []}))
    topics = load_taxonomy("config/taxonomy.yaml")
    llm = MagicMock()
    # Default LLM responses — tests can override per-test
    llm.chat_text.return_value = "How was X?"
    llm.embed.return_value = [0.1] * 1536
    llm.chat_json.return_value = {t.topic_id: {"mentioned": False, "sentiment": None, "assertion": None}
                                  for t in topics}
    flow = AskFlow(
        repo=repo, llm=llm,
        taxonomy=topics,
        topic_embeddings={},
        field_cluster=build_field_cluster_map(topics),
        weights_path="config/weights.yaml",
    )
    return flow, repo, llm


def test_flow_submit_review_returns_questions(tmp_path):
    flow, repo, llm = _setup_flow(tmp_path)
    today = date(2025, 9, 1)
    questions = flow.submit_review("p1", "", today)  # empty review, cold start
    assert 1 <= len(questions) <= 2
    for q in questions:
        assert q.field_id  # has a field
        assert q.question_text  # has rendered text


def test_flow_empty_review_skips_translate_and_embed(tmp_path):
    flow, repo, llm = _setup_flow(tmp_path)
    flow.submit_review("p1", "", date(2025, 9, 1))
    # For cold start with empty review: no translate, no embed, no tag (tag_review short-circuits on None)
    # But render_question IS called on each picked field.
    llm.embed.assert_not_called()


def test_flow_submit_review_persists_with_live_source(tmp_path):
    flow, repo, llm = _setup_flow(tmp_path)
    flow.submit_review("p1", "Great stay", date(2025, 9, 1))
    reviews = repo.list_reviews_for("p1")
    assert len(reviews) == 1
    assert reviews[0].source == "live"


def test_flow_submit_answer_scored_updates_field_state(tmp_path):
    flow, repo, llm = _setup_flow(tmp_path)
    questions = flow.submit_review("p1", "", date(2025, 9, 1))
    q = questions[0]
    # Answer with a simple rating
    if q.input_type == "rating_1_5":
        ans_text = "4 out of 5"
    elif q.input_type == "yes_no":
        ans_text = "yes"
    else:
        llm.chat_json.return_value = {"value": "some useful info", "abstain": False}
        ans_text = "The wifi was really fast everywhere"
    answer = flow.submit_answer("p1", q, ans_text, date(2025, 9, 1))
    assert answer.status == "scored"
    # Field state was updated
    fs = repo.get_field_state("p1", q.field_id)
    assert fs.value_known is True
    assert fs.mention_count >= 1
    assert fs.last_confirmed_date == date(2025, 9, 1)


def test_flow_submit_answer_without_prior_review_raises(tmp_path):
    flow, repo, llm = _setup_flow(tmp_path)
    q = Question(field_id="rating:checkin", question_text="How was check-in?",
                 input_type="rating_1_5", reason="none")
    try:
        flow.submit_answer("p1", q, "4", date(2025, 9, 1))
        raise AssertionError("should have raised")
    except RuntimeError as e:
        assert "No pending review" in str(e)


def test_flow_submit_answer_skipped_does_not_update_state(tmp_path):
    flow, repo, llm = _setup_flow(tmp_path)
    questions = flow.submit_review("p1", "", date(2025, 9, 1))
    q = questions[0]
    # Get pre-state
    before = repo.get_field_state("p1", q.field_id)
    answer = flow.submit_answer("p1", q, None, date(2025, 9, 1))  # None = skip
    assert answer.status == "skipped"
    after = repo.get_field_state("p1", q.field_id)
    # mention_count and value_known unchanged by a skipped answer
    assert after.value_known == before.value_known
    assert after.mention_count == before.mention_count
