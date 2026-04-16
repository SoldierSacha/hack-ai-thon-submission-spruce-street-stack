from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.db import Repo
from src.enrich import enrich_review
from src.ingest import load_properties, load_reviews
from src.llm import LlmClient
from src.signals import build_all_field_states
from src.taxonomy import load_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[2]


def build(
    descriptions_csv: str | Path = REPO_ROOT / "hackathon resources" / "Description_PROC.csv",
    reviews_csv: str | Path = REPO_ROOT / "hackathon resources" / "Reviews_PROC.csv",
    taxonomy_yaml: str | Path = Path(__file__).resolve().parents[1] / "config" / "taxonomy.yaml",
    db_path: str | Path = Path(__file__).resolve().parents[1] / "data" / "state.sqlite",
    limit_reviews: int | None = None,
    max_workers: int = 8,
    cache_dir: str | Path = Path(__file__).resolve().parents[1] / "data" / "cache",
) -> None:
    """
    Full build pipeline. Idempotent — safe to re-run (LLM cache makes re-runs cheap).

    Args:
      limit_reviews: if set, enrich only the first N reviews (for smoke testing).
      max_workers: parallel threads for LLM calls. 8 is safe for gpt-4.1-mini RPM.
    """
    print(f"[build] db={db_path}, cache={cache_dir}")
    repo = Repo(db_path)
    repo.init_schema()

    print(f"[build] loading properties from {descriptions_csv}")
    properties = load_properties(descriptions_csv)
    for p in properties:
        repo.upsert_property(p)
    print(f"[build] upserted {len(properties)} properties")

    print(f"[build] loading reviews from {reviews_csv}")
    reviews = load_reviews(reviews_csv)
    if limit_reviews is not None:
        reviews = reviews[:limit_reviews]
    print(f"[build] {len(reviews)} reviews to enrich")

    topics = load_taxonomy(taxonomy_yaml)
    llm = LlmClient(cache_dir=cache_dir)

    done = 0
    failed = 0
    start = time.monotonic()

    def _enrich_only(rev):
        # Pure I/O-bound LLM work; no DB access (SQLite connection is main-thread only).
        return rev, enrich_review(rev, topics, llm)

    def _persist(rev, result):
        text_en, lang, emb, tags = result
        rev.review_text_en = text_en
        rev.lang = lang
        repo.upsert_review(rev)
        if emb is not None:
            repo.set_embedding(rev.review_id, emb)
        tag_list = [
            {
                "field_id": f"topic:{tid}",
                "mentioned": bool(v.get("mentioned")),
                "sentiment": v.get("sentiment"),
                "assertion": v.get("assertion"),
            }
            for tid, v in tags.items()
        ]
        if tag_list:
            repo.upsert_review_tags(rev.review_id, tag_list)

    print(f"[build] enriching with {max_workers} workers…")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_enrich_only, rev): rev for rev in reviews}
        for fut in as_completed(futures):
            rev = futures[fut]
            try:
                _, result = fut.result()
                _persist(rev, result)
                done += 1
            except Exception as e:
                failed += 1
                print(f"  [!] {rev.review_id} failed: {e}")
            if (done + failed) % 50 == 0:
                elapsed = time.monotonic() - start
                rate = (done + failed) / elapsed if elapsed > 0 else 0
                eta = (len(reviews) - done - failed) / rate if rate > 0 else 0
                print(f"  [{done + failed}/{len(reviews)}] ok={done} fail={failed} "
                      f"rate={rate:.1f}/s eta={eta:.0f}s")

    elapsed = time.monotonic() - start
    print(f"[build] done in {elapsed:.1f}s. ok={done} fail={failed}")
    if failed > 0:
        print(f"[build] WARNING: {failed} reviews failed enrichment. Rerun to retry (cache will skip successful ones).")

    print("[build] aggregating field states...")
    n = build_all_field_states(repo, taxonomy_yaml)
    print(f"[build] wrote {n} field states")


def run_ask(property_id: str, review_text: str, today_override: str | None = None) -> None:
    """
    Submit a review, show the follow-up question(s), read typed answers on stdin, persist.
    Press Enter (empty) to skip.
    """
    from datetime import date
    from src.db import Repo
    from src.llm import LlmClient
    from src.taxonomy import load_taxonomy
    from src.ranker import build_field_cluster_map
    from src.flow import AskFlow

    db_path = Path(__file__).resolve().parents[1] / "data" / "state.sqlite"
    taxonomy_yaml = Path(__file__).resolve().parents[1] / "config" / "taxonomy.yaml"
    cache_dir = Path(__file__).resolve().parents[1] / "data" / "cache"

    if property_id == "list":
        # List properties as a dev helper.
        repo = Repo(db_path)
        for p in repo.list_properties():
            print(f"  {p.eg_property_id}  {p.city}, {p.country}  {p.star_rating or '?'}-star")
        return

    repo = Repo(db_path)
    llm = LlmClient(cache_dir=cache_dir)
    topics = load_taxonomy(taxonomy_yaml)

    today = date.fromisoformat(today_override) if today_override else _max_review_date(repo)

    flow = AskFlow(
        repo=repo,
        llm=llm,
        taxonomy=topics,
        topic_embeddings={},
        field_cluster=build_field_cluster_map(topics),
    )

    prop = repo.get_property(property_id)
    if prop is None:
        print(f"[ask] Unknown property_id {property_id}. Try `run.py ask list`.")
        return

    print(f"[ask] Property: {prop.city}, {prop.country} ({prop.star_rating or '?'}\u2605)")
    print(f"[ask] Today (data-relative): {today}")
    print(f"[ask] Review: {review_text!r}")
    print()
    print("[ask] Picking follow-up questions...")

    questions = flow.submit_review(property_id, review_text, today=today)

    if not questions:
        print("  (no follow-up needed \u2014 property is fully covered)")
        return

    for i, q in enumerate(questions, start=1):
        print()
        print(f"  Question {i}: {q.question_text}")
        print(f"    (why: {q.reason})")
        print(f"    (input type: {q.input_type})")
        try:
            ans = input("    Your answer (Enter to skip): ").strip()
        except EOFError:
            ans = ""
        ans_to_pass = ans if ans else None
        answer = flow.submit_answer(property_id, q, ans_to_pass, today=today)
        print(f"    \u2192 status={answer.status}, parsed_value={answer.parsed_value!r}")

    # Summary
    print()
    print("[ask] Updated field states:")
    for q in questions:
        fs = repo.get_field_state(property_id, q.field_id)
        if fs is not None:
            print(f"  {fs.field_id}: value_known={fs.value_known}, "
                  f"mention_count={fs.mention_count}, last_confirmed={fs.last_confirmed_date}")


def _max_review_date(repo) -> "date":
    """Find the newest acquisition_date across all reviews (reproducible 'today')."""
    from datetime import date
    all_dates = []
    for p in repo.list_properties():
        for r in repo.list_reviews_for(p.eg_property_id):
            all_dates.append(r.acquisition_date)
    return max(all_dates) if all_dates else date.today()
