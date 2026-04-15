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
