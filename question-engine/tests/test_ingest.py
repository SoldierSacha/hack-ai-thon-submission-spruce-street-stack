from src.ingest import load_properties

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
