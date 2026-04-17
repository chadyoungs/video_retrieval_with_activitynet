#!/usr/bin/env python3
import sqlite3
from typing import Dict, List

DB_NAME = "./database/video_metadata.db"


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        DROP TABLE IF EXISTS video_clips
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS video_clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_file_name TEXT NOT NULL,
            video_file_path TEXT NOT NULL,
            segment_start FLOAT,
            segment_end FLOAT,
            scene_env TEXT,
            scene_type TEXT,
            weather TEXT,
            lighting TEXT,
            time_of_day TEXT,
            person_count TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    # Column names are validated against a fixed allowlist to prevent any
    # accidental SQL injection if this tuple is ever widened.
    _ANNOTATION_COLUMNS = frozenset(
        (
            "scene_env",
            "scene_type",
            "weather",
            "lighting",
            "time_of_day",
            "person_count",
        )
    )
    for col in (
        "scene_env",
        "scene_type",
        "weather",
        "lighting",
        "time_of_day",
        "person_count",
    ):
        assert col in _ANNOTATION_COLUMNS, f"Unexpected column name: {col}"
        c.execute(f"CREATE INDEX IF NOT EXISTS idx_{col} ON video_clips({col})")
    conn.commit()
    conn.close()


def save_to_db(video_file_name, video_file_path, segment_start, segment_end, anno):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO video_clips (
            video_file_name, video_file_path, segment_start, segment_end, scene_env, scene_type, weather, lighting,
            time_of_day, person_count
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
    """,
        (
            video_file_name,
            video_file_path,
            segment_start,
            segment_end,
            anno.get("scene_env"),
            anno.get("scene_type"),
            anno.get("weather"),
            anno.get("lighting"),
            anno.get("time_of_day"),
            anno.get("person_count"),
        ),
    )
    conn.commit()
    conn.close()


def batch_insert_sqlite(batch_data):
    conn = get_db_connection()
    if not batch_data:
        return "Empty batch"

    cursor = conn.cursor()

    conn.isolation_level = None

    cursor.execute("BEGIN TRANSACTION")
    try:
        sql = """
        INSERT INTO video_clips (
            video_file_name, video_file_path, segment_start, segment_end, scene_env, scene_type, weather, lighting,
            time_of_day, person_count
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """
        formatted_data = []
        for item in batch_data:
            name, path, s, e, ann_dict = item
            formatted_data.append(
                (
                    name,
                    path,
                    s,
                    e,
                    ann_dict.get("scene_env"),
                    ann_dict.get("scene_type"),
                    ann_dict.get("weather"),
                    ann_dict.get("lighting"),
                    ann_dict.get("time_of_day"),
                    ann_dict.get("person_count"),
                )
            )
        cursor.executemany(sql, formatted_data)
        conn.commit()
        return f"Batch inserted: {len(batch_data)} rows"
    except Exception as e:
        conn.rollback()
        return f"Batch failed: {e}"
    finally:
        cursor.close()
        conn.close()


def query_annotation_by_conditions(conditions: Dict, limit: int = 5) -> List[Dict]:
    """
    Return video clips that match *any* of the supplied annotation conditions,
    scored by how many conditions they satisfy.  Results are ordered by match
    count (descending) so that rows matching all conditions rank above partial
    matches.  This replaces the previous AND-only query that provided no
    ranking differentiation among results.
    """
    if not conditions:
        return []

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Build a CASE expression that adds 1 for each matching condition so
        # that the score is proportional to the number of satisfied conditions.
        score_cases = []
        score_params = []
        where_clauses = []
        where_params = []

        for key, value in conditions.items():
            score_cases.append(f"CASE WHEN {key} = ? THEN 1 ELSE 0 END")
            score_params.append(value)
            where_clauses.append(f"{key} = ?")
            where_params.append(value)

        score_expr = " + ".join(score_cases)
        where_expr = " OR ".join(where_clauses)

        query_sql = f"""
            SELECT video_file_name, video_file_path, segment_start, segment_end,
                   scene_env, scene_type, weather, lighting, time_of_day, person_count,
                   ({score_expr}) AS match_count
            FROM video_clips
            WHERE {where_expr}
            ORDER BY match_count DESC
            LIMIT ?
        """
        params = score_params + where_params + [limit]

        cursor.execute(query_sql, params)
        rows = cursor.fetchall()

        n_conditions = len(conditions)
        results = []
        for row in rows:
            results.append(
                {
                    "video_file_name": row["video_file_name"],
                    "video_file_path": row["video_file_path"],
                    "segment_start": row["segment_start"],
                    "segment_end": row["segment_end"],
                    "scene_env": row["scene_env"],
                    "scene_type": row["scene_type"],
                    "weather": row["weather"],
                    "lighting": row["lighting"],
                    "time_of_day": row["time_of_day"],
                    "person_count": row["person_count"],
                    # Normalise to [0, 1] so that hybrid_retrieval weight_sql is meaningful.
                    "_match_count": row["match_count"],
                    "_n_conditions": n_conditions,
                }
            )
        return results
    except Exception as e:
        print(f"SQL query error: {e}")
        return []
    finally:
        if conn:
            conn.close()


def search_sql(annotation_conditions: Dict, limit: int = 5) -> List[Dict]:
    try:
        sql_hits = query_annotation_by_conditions(
            conditions=annotation_conditions, limit=limit
        )
        n = len(annotation_conditions)
        for hit in sql_hits:
            matched = hit.pop("_match_count", n)
            hit.pop("_n_conditions", None)
            # Score proportional to the fraction of conditions satisfied so
            # that partial matches rank lower than full matches in the hybrid
            # fusion step.
            hit["score"] = matched / n if n > 0 else 1.0
        return sql_hits
    except Exception as e:
        print(f"SQL search error: {e}")
        return []


if __name__ == "__main__":
    init_db()
