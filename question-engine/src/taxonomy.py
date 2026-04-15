from pathlib import Path
import yaml
from src.models import TaxonomyTopic, SUB_RATING_KEYS

SCHEMA_DESCRIPTION_FIELDS = (
    "pet_policy", "children_and_extra_bed_policy", "check_in_instructions",
    "know_before_you_go", "check_out_policy",
    "property_amenity_accessibility", "property_amenity_activities_nearby",
    "property_amenity_business_services", "property_amenity_food_and_drink",
    "property_amenity_outdoor", "property_amenity_spa",
    "property_amenity_things_to_do",
)

def load_taxonomy(path: str | Path) -> list[TaxonomyTopic]:
    data = yaml.safe_load(Path(path).read_text())
    return [TaxonomyTopic(
        topic_id=t["id"], label=t["label"],
        cluster_id=t["cluster"], question_hint=t["question_hint"],
    ) for t in data["topics"]]

def schema_field_ids() -> list[str]:
    rating_ids = [f"rating:{k}" for k in SUB_RATING_KEYS]
    desc_ids = [f"schema:{f}" for f in SCHEMA_DESCRIPTION_FIELDS]
    return rating_ids + desc_ids

def all_field_ids(tax_path: str | Path = "config/taxonomy.yaml") -> list[str]:
    return schema_field_ids() + [f"topic:{t.topic_id}" for t in load_taxonomy(tax_path)]
