from datetime import date
from src.ingest import load_properties, load_reviews

def test_load_properties_parses_amenity_json():
    props = load_properties("tests/fixtures/descriptions_tiny.csv")
    assert len(props) == 5
    p0 = props[0]
    assert "internet" in p0.amenities
    assert any("wifi" in a.lower() for a in p0.amenities["internet"])

def test_load_properties_handles_missing_star_rating():
    props = load_properties("tests/fixtures/descriptions_tiny.csv")
    # at least one of the 5 fixtures has a NaN star_rating
    assert any(p.star_rating is None for p in props)

def test_load_reviews_parses_date_and_rating():
    reviews = load_reviews("tests/fixtures/reviews_tiny.csv")
    assert len(reviews) == 10
    r0 = reviews[0]
    assert isinstance(r0.acquisition_date, date)
    # 0 in rating JSON must become None
    for k in ("checkin", "location", "onlinelisting"):
        assert getattr(r0.rating, k) is None

def test_load_reviews_assigns_unique_ids():
    reviews = load_reviews("tests/fixtures/reviews_tiny.csv")
    assert len({r.review_id for r in reviews}) == len(reviews)
