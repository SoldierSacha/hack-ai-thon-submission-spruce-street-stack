from datetime import date
from pathlib import Path
import sqlite3
import numpy as np
from src.db import Repo
from src.models import Property, Review, RatingBreakdown, FieldState, Answer


def _make_review(rid: str = "p1:0") -> Review:
    return Review(
        review_id=rid,
        eg_property_id="p1",
        acquisition_date=date(2025, 9, 1),
        rating=RatingBreakdown(overall=5),
        review_text_orig="great",
    )


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


def test_embedding_roundtrip(tmp_db):
    repo = Repo(tmp_db)
    repo.init_schema()

    r = _make_review()
    repo.upsert_review(r)

    vec = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    repo.set_embedding(r.review_id, vec)

    loaded = repo.load_embedding(r.review_id)
    np.testing.assert_array_equal(loaded, vec)


def test_upsert_review_preserves_embedding(tmp_db):
    repo = Repo(tmp_db)
    repo.init_schema()

    r = _make_review()
    repo.upsert_review(r)

    vec = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    repo.set_embedding(r.review_id, vec)

    # Simulate pipeline re-run.
    repo.upsert_review(r)

    loaded = repo.load_embedding(r.review_id)
    np.testing.assert_array_equal(loaded, vec)


def test_upsert_review_tags_and_append_only_answers(tmp_db):
    repo = Repo(tmp_db)
    repo.init_schema()

    r = _make_review()
    repo.upsert_review(r)

    tags = [
        {"field_id": "topic:wifi", "mentioned": True, "sentiment": 1, "assertion": "fast"},
        {"field_id": "topic:breakfast", "mentioned": True, "sentiment": 0, "assertion": None},
    ]
    repo.upsert_review_tags(r.review_id, tags)

    answer = Answer(
        field_id="topic:wifi",
        question_text="How was the wifi?",
        answer_text="Great",
        status="scored",
    )
    repo.record_answer(r.review_id, answer)

    conn = sqlite3.connect(tmp_db)
    try:
        tag_count = conn.execute(
            "SELECT COUNT(*) FROM review_tags WHERE review_id = ?", (r.review_id,)
        ).fetchone()[0]
        assert tag_count == len(tags)

        answer_count = conn.execute(
            "SELECT COUNT(*) FROM answers WHERE review_id = ?", (r.review_id,)
        ).fetchone()[0]
        assert answer_count == 1
    finally:
        conn.close()

    # Append-only: a second record_answer must NOT replace the prior row.
    repo.record_answer(r.review_id, answer)

    conn = sqlite3.connect(tmp_db)
    try:
        answer_count = conn.execute(
            "SELECT COUNT(*) FROM answers WHERE review_id = ?", (r.review_id,)
        ).fetchone()[0]
        assert answer_count == 2
    finally:
        conn.close()


def test_list_review_tags_for_all_roundtrip(tmp_path):
    db = tmp_path / "state.sqlite"
    repo = Repo(db)
    repo.init_schema()
    # Seed a property and review so the FK-less tags have a review to anchor to
    repo.upsert_property(Property(eg_property_id="p1"))
    repo.upsert_review(Review(review_id="p1:0", eg_property_id="p1",
                              acquisition_date=date(2024, 1, 1),
                              rating=RatingBreakdown()))
    repo.upsert_review_tags("p1:0", [
        {"field_id": "topic:wifi", "mentioned": True, "sentiment": 1, "assertion": "fast wifi"},
        {"field_id": "topic:pool", "mentioned": False, "sentiment": None, "assertion": None},
    ])
    tags = repo.list_review_tags_for_all()
    assert "p1:0" in tags
    assert len(tags["p1:0"]) == 2
    wifi = next(t for t in tags["p1:0"] if t["field_id"] == "topic:wifi")
    assert wifi["mentioned"] is True
    assert wifi["sentiment"] == 1
    assert wifi["assertion"] == "fast wifi"
