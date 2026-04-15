from src.taxonomy import load_taxonomy, schema_field_ids, all_field_ids

def test_load_taxonomy_has_topics():
    topics = load_taxonomy("config/taxonomy.yaml")
    assert len(topics) >= 20
    assert all(t.topic_id and t.label and t.cluster_id for t in topics)

def test_schema_fields_include_empty_subratings():
    fids = schema_field_ids()
    assert "rating:checkin" in fids
    assert "schema:pet_policy" in fids
    assert "schema:property_amenity_spa" in fids

def test_all_field_ids_is_union():
    assert set(all_field_ids()) == set(schema_field_ids()) | {f"topic:{t.topic_id}" for t in load_taxonomy("config/taxonomy.yaml")}
