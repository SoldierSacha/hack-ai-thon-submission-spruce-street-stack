from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from src.models import Property, Review, FieldState, Answer

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS properties (
    eg_property_id TEXT PRIMARY KEY,
    raw_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    eg_property_id TEXT NOT NULL,
    acquisition_date TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_reviews_prop ON reviews(eg_property_id);
CREATE TABLE IF NOT EXISTS review_tags (
    review_id TEXT NOT NULL,
    field_id TEXT NOT NULL,
    mentioned INTEGER NOT NULL,
    sentiment INTEGER,
    assertion TEXT,
    PRIMARY KEY (review_id, field_id)
);
CREATE TABLE IF NOT EXISTS field_state (
    eg_property_id TEXT NOT NULL,
    field_id TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    PRIMARY KEY (eg_property_id, field_id)
);
CREATE TABLE IF NOT EXISTS taxonomy (
    topic_id TEXT PRIMARY KEY,
    raw_json TEXT NOT NULL,
    embedding BLOB
);
CREATE TABLE IF NOT EXISTS answers (
    answer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id TEXT NOT NULL,
    field_id TEXT NOT NULL,
    raw_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Repo:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)

    # --- properties ---

    def upsert_property(self, p: Property) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO properties(eg_property_id, raw_json) VALUES (?, ?)",
                (p.eg_property_id, p.model_dump_json()),
            )

    def get_property(self, pid: str) -> Property | None:
        row = self._conn.execute(
            "SELECT raw_json FROM properties WHERE eg_property_id = ?", (pid,)
        ).fetchone()
        return Property.model_validate_json(row["raw_json"]) if row else None

    def list_properties(self) -> list[Property]:
        rows = self._conn.execute(
            "SELECT raw_json FROM properties ORDER BY eg_property_id"
        ).fetchall()
        return [Property.model_validate_json(r["raw_json"]) for r in rows]

    # --- reviews ---

    def upsert_review(self, r: Review) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO reviews(review_id, eg_property_id, acquisition_date, raw_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(review_id) DO UPDATE SET
                    eg_property_id = excluded.eg_property_id,
                    acquisition_date = excluded.acquisition_date,
                    raw_json = excluded.raw_json""",
                (
                    r.review_id,
                    r.eg_property_id,
                    r.acquisition_date.isoformat(),
                    r.model_dump_json(),
                ),
            )

    def set_embedding(self, review_id: str, embedding: np.ndarray) -> None:
        if embedding.ndim != 1:
            raise ValueError(f"embedding must be 1-D, got shape {embedding.shape}")
        blob = embedding.astype(np.float32, copy=False).tobytes()
        with self._conn:
            cur = self._conn.execute(
                "UPDATE reviews SET embedding = ? WHERE review_id = ?",
                (blob, review_id),
            )
            if cur.rowcount == 0:
                raise KeyError(review_id)

    def list_reviews_for(self, pid: str) -> list[Review]:
        rows = self._conn.execute(
            "SELECT raw_json FROM reviews WHERE eg_property_id = ? ORDER BY acquisition_date",
            (pid,),
        ).fetchall()
        return [Review.model_validate_json(r["raw_json"]) for r in rows]

    def list_reviews(self) -> list[Review]:
        rows = self._conn.execute(
            "SELECT raw_json FROM reviews ORDER BY eg_property_id, acquisition_date"
        ).fetchall()
        return [Review.model_validate_json(r["raw_json"]) for r in rows]

    def load_embedding(self, review_id: str) -> np.ndarray | None:
        row = self._conn.execute(
            "SELECT embedding FROM reviews WHERE review_id = ?", (review_id,)
        ).fetchone()
        if row is None or row["embedding"] is None:
            return None
        return np.frombuffer(row["embedding"], dtype=np.float32)

    # --- review_tags ---

    def upsert_review_tags(self, review_id: str, tags: list[dict]) -> None:
        with self._conn:
            for tag in tags:
                self._conn.execute(
                    "INSERT OR REPLACE INTO review_tags(review_id, field_id, mentioned, sentiment, assertion) VALUES (?, ?, ?, ?, ?)",
                    (
                        review_id,
                        tag["field_id"],
                        1 if tag["mentioned"] else 0,
                        tag.get("sentiment"),
                        tag.get("assertion"),
                    ),
                )

    def list_review_tags_all(self) -> dict[str, list[dict]]:
        """
        Return every review tag, grouped by review_id.
        Each tag dict: {field_id, mentioned, sentiment, assertion}.
        """
        rows = self._conn.execute(
            "SELECT review_id, field_id, mentioned, sentiment, assertion FROM review_tags"
        ).fetchall()
        out: dict[str, list[dict]] = {}
        for row in rows:
            out.setdefault(row["review_id"], []).append({
                "field_id": row["field_id"],
                "mentioned": bool(row["mentioned"]),
                "sentiment": row["sentiment"],
                "assertion": row["assertion"],
            })
        return out

    def list_review_tags_for_all(self) -> dict[str, list[dict]]:
        """
        Return all review_tags grouped by review_id.
        Each list item has keys: field_id, mentioned (bool), sentiment (int|None), assertion (str|None).
        """
        out: dict[str, list[dict]] = {}
        rows = self._conn.execute(
            "SELECT review_id, field_id, mentioned, sentiment, assertion FROM review_tags"
        ).fetchall()
        for row in rows:
            out.setdefault(row["review_id"], []).append({
                "field_id": row["field_id"],
                "mentioned": bool(row["mentioned"]),
                "sentiment": row["sentiment"],
                "assertion": row["assertion"],
            })
        return out

    # --- field_state ---

    def upsert_field_state(self, fs: FieldState) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO field_state(eg_property_id, field_id, raw_json) VALUES (?, ?, ?)",
                (fs.eg_property_id, fs.field_id, fs.model_dump_json()),
            )

    def get_field_state(self, pid: str, fid: str) -> FieldState | None:
        row = self._conn.execute(
            "SELECT raw_json FROM field_state WHERE eg_property_id = ? AND field_id = ?",
            (pid, fid),
        ).fetchone()
        return FieldState.model_validate_json(row["raw_json"]) if row else None

    def list_field_states_for(self, pid: str) -> list[FieldState]:
        rows = self._conn.execute(
            "SELECT raw_json FROM field_state WHERE eg_property_id = ? ORDER BY field_id",
            (pid,),
        ).fetchall()
        return [FieldState.model_validate_json(r["raw_json"]) for r in rows]

    # --- answers (append-only) ---

    def record_answer(self, review_id: str, answer: Answer) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO answers(review_id, field_id, raw_json) VALUES (?, ?, ?)",
                (review_id, answer.field_id, answer.model_dump_json()),
            )

    # --- llm_cache ---

    def cache_get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT response FROM llm_cache WHERE key = ?", (key,)
        ).fetchone()
        return row["response"] if row else None

    def cache_put(self, key: str, response: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache(key, response, created_at) VALUES (?, ?, ?)",
                (key, response, ts),
            )
