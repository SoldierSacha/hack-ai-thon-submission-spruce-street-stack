from datetime import date
from src.db import Repo
from src.models import Property, Review, RatingBreakdown, FieldState

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
