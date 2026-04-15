#!/usr/bin/env python3
import json
import sqlite3
from typing import Dict, List

DB_NAME = "./database/video_metadata.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        DROP TABLE IF EXISTS video_clips
    """)
    c.execute("""
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
    """)
    conn.commit()
    conn.close()


def save_to_db(video_file_name, video_file_path, segment_start, segment_end, anno):
    conn = sqlite3.connect(DB_NAME)
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
    if not batch_data:
        return "Empty batch"

    conn = sqlite3.connect(DB_NAME)
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


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def query_annotation_by_conditions(conditions: Dict, limit: int = 5) -> List[Dict]:
    if not conditions:
        return []

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        where_clauses = []
        params = []
        for key, value in conditions.items():
            where_clauses.append(f"{key} = ?")
            params.append(value)

        query_sql = f"""
            SELECT video_file_name, video_file_path, segment_start, segment_end,
                   scene_env, scene_type, weather, lighting, time_of_day, person_count
            FROM video_clips
            WHERE {' AND '.join(where_clauses)}
            LIMIT ?
        """
        params.append(limit)

        cursor.execute(query_sql, params)
        rows = cursor.fetchall()

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
        for hit in sql_hits:
            hit["score"] = 1.0
        return sql_hits
    except Exception as e:
        print(f"SQL search error: {e}")
        return []


if __name__ == "__main__":
    init_db()
