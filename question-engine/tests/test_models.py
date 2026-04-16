from datetime import date
from src.models import Property, Review, FieldState, RatingBreakdown, TaxonomyTopic, ScoredField, EnrichmentMeta, SubmitResult, Question

def test_rating_breakdown_zero_is_none():
    r = RatingBreakdown.from_raw({"overall": 5, "checkin": 0, "service": 4})
    assert r.overall == 5
    assert r.checkin is None          # 0 in data means NULL
    assert r.service == 4

def test_rating_breakdown_accepts_float_values():
    """Real CSV stores ratings as floats (5.0, not 5). from_raw must coerce."""
    r = RatingBreakdown.from_raw({"overall": 5.0, "service": 4.0, "checkin": 0.0})
    assert r.overall == 5         # int, not float
    assert r.service == 4
    assert r.checkin is None      # 0.0 → None

def test_rating_breakdown_rejects_bool_values():
    """bool is a subclass of int in Python; guard against it."""
    r = RatingBreakdown.from_raw({"overall": True, "service": 5})
    assert r.overall is None      # True must not leak through
    assert r.service == 5

def test_property_accepts_missing_star_rating():
    p = Property(eg_property_id="abc", city="Pompei", country="Italy",
                 star_rating=None, guestrating_avg_expedia=8.4)
    assert p.star_rating is None

def test_field_state_ema_can_be_none_when_sparse():
    fs = FieldState(eg_property_id="abc", field_id="topic:wifi",
                    value_known=True, mention_count=2)
    assert fs.short_ema is None
    assert fs.long_ema is None

def test_scored_field_stores_component_scores():
    fs = FieldState(eg_property_id="p1", field_id="topic:wifi", value_known=False)
    sf = ScoredField(
        field_state=fs, composite=0.72,
        missing=1.0, stale=0.0, coverage=0.85, redundancy=0.12,
        rank=1, cluster="connectivity",
    )
    assert sf.composite == 0.72
    assert sf.missing == 1.0
    assert sf.rank == 1
    assert sf.cluster == "connectivity"
    assert sf.field_state.field_id == "topic:wifi"

def test_submit_result_bundles_questions_and_metadata():
    meta = EnrichmentMeta(lang="de", translated=True, embedding_dim=384, topics_tagged=8, topics_total=28)
    q = Question(field_id="rating:checkin", question_text="How was check-in?",
                 input_type="rating_1_5", reason="No data")
    fs = FieldState(eg_property_id="p1", field_id="rating:checkin", value_known=False)
    sf = ScoredField(field_state=fs, composite=0.72, missing=1.0, stale=0.0,
                     coverage=0.85, redundancy=0.0, rank=1, cluster="service")
    result = SubmitResult(questions=[q], scored_fields=[sf], enrichment=meta, total_fields=58)
    assert len(result.questions) == 1
    assert result.enrichment.lang == "de"
    assert result.enrichment.translated is True
    assert result.total_fields == 58
