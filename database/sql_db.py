#!/usr/bin/env python3
import sqlite3
from typing import Dict, List

DB_NAME = "./database/video_metadata.db"

ANNOTATION_FIELDS = (
    "scene_env",
    "scene_type",
    "weather",
    "lighting",
    "time_of_day",
    "person_count",
    "driving_context",
    "road_user_density",
    "traffic_flow",
)

METADATA_FILTER_FIELDS = (
    "dataset_name",
    "dataset_version",
    "dataset_split",
    "camera_channel",
    "clip_id",
    "scene_token",
    "location",
)

ALLOWED_FILTER_FIELDS = frozenset(ANNOTATION_FIELDS + METADATA_FILTER_FIELDS)


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
            dataset_name TEXT,
            dataset_version TEXT,
            dataset_split TEXT,
            clip_id TEXT,
            segment_id TEXT,
            scene_token TEXT,
            sample_start_token TEXT,
            sample_end_token TEXT,
            camera_channel TEXT,
            log_token TEXT,
            location TEXT,
            map_token TEXT,
            clip_start_ts INTEGER,
            clip_end_ts INTEGER,
            clip_fps FLOAT,
            clip_frame_count INTEGER,
            segment_start FLOAT,
            segment_end FLOAT,
            scene_env TEXT,
            scene_type TEXT,
            weather TEXT,
            lighting TEXT,
            time_of_day TEXT,
            person_count TEXT,
            driving_context TEXT,
            road_user_density TEXT,
            traffic_flow TEXT,
            semantic_summary TEXT,
            embedding_model TEXT,
            annotation_model TEXT,
            processing_ts TEXT,
            quality_flag TEXT,
            ego_avg_speed_mps FLOAT,
            ego_max_speed_mps FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    for col in ALLOWED_FILTER_FIELDS:
        c.execute(f"CREATE INDEX IF NOT EXISTS idx_{col} ON video_clips({col})")
    c.execute("CREATE INDEX IF NOT EXISTS idx_video_segment ON video_clips(video_file_name, segment_start)")
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_dataset_camera_scene "
        "ON video_clips(dataset_name, camera_channel, scene_token)"
    )
    conn.commit()
    conn.close()


def _normalize_record(item):
    if isinstance(item, dict):
        ann = item.get("annotation", {})
        metadata = item.get("metadata", {})
        return {
            "video_file_name": item.get("video_file_name"),
            "video_file_path": item.get("video_file_path"),
            "segment_start": item.get("segment_start"),
            "segment_end": item.get("segment_end"),
            "dataset_name": metadata.get("dataset_name", "activitynet"),
            "dataset_version": metadata.get("dataset_version", "unknown"),
            "dataset_split": metadata.get("dataset_split", "unknown"),
            "clip_id": metadata.get("clip_id"),
            "segment_id": metadata.get("segment_id"),
            "scene_token": metadata.get("scene_token"),
            "sample_start_token": metadata.get("sample_start_token"),
            "sample_end_token": metadata.get("sample_end_token"),
            "camera_channel": metadata.get("camera_channel", "unknown"),
            "log_token": metadata.get("log_token"),
            "location": metadata.get("location"),
            "map_token": metadata.get("map_token"),
            "clip_start_ts": metadata.get("clip_start_ts"),
            "clip_end_ts": metadata.get("clip_end_ts"),
            "clip_fps": metadata.get("clip_fps"),
            "clip_frame_count": metadata.get("clip_frame_count"),
            "scene_env": ann.get("scene_env"),
            "scene_type": ann.get("scene_type"),
            "weather": ann.get("weather"),
            "lighting": ann.get("lighting"),
            "time_of_day": ann.get("time_of_day"),
            "person_count": ann.get("person_count"),
            "driving_context": ann.get("driving_context"),
            "road_user_density": ann.get("road_user_density"),
            "traffic_flow": ann.get("traffic_flow"),
            "semantic_summary": ann.get("semantic_summary"),
            "embedding_model": metadata.get("embedding_model"),
            "annotation_model": metadata.get("annotation_model"),
            "processing_ts": metadata.get("processing_ts"),
            "quality_flag": metadata.get("quality_flag", "ok"),
            "ego_avg_speed_mps": metadata.get("ego_avg_speed_mps"),
            "ego_max_speed_mps": metadata.get("ego_max_speed_mps"),
        }

    # backward compatibility: (name, path, s, e, ann_dict)
    name, path, s, e, ann_dict = item
    return {
        "video_file_name": name,
        "video_file_path": path,
        "segment_start": s,
        "segment_end": e,
        "dataset_name": "activitynet",
        "dataset_version": "unknown",
        "dataset_split": "unknown",
        "clip_id": None,
        "segment_id": None,
        "scene_token": None,
        "sample_start_token": None,
        "sample_end_token": None,
        "camera_channel": "unknown",
        "log_token": None,
        "location": None,
        "map_token": None,
        "clip_start_ts": None,
        "clip_end_ts": None,
        "clip_fps": None,
        "clip_frame_count": None,
        "scene_env": ann_dict.get("scene_env"),
        "scene_type": ann_dict.get("scene_type"),
        "weather": ann_dict.get("weather"),
        "lighting": ann_dict.get("lighting"),
        "time_of_day": ann_dict.get("time_of_day"),
        "person_count": ann_dict.get("person_count"),
        "driving_context": ann_dict.get("driving_context"),
        "road_user_density": ann_dict.get("road_user_density"),
        "traffic_flow": ann_dict.get("traffic_flow"),
        "semantic_summary": ann_dict.get("semantic_summary"),
        "embedding_model": None,
        "annotation_model": None,
        "processing_ts": None,
        "quality_flag": "ok",
        "ego_avg_speed_mps": None,
        "ego_max_speed_mps": None,
    }


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
            video_file_name, video_file_path, dataset_name, dataset_version, dataset_split,
            clip_id, segment_id, scene_token, sample_start_token, sample_end_token, camera_channel,
            log_token, location, map_token, clip_start_ts, clip_end_ts, clip_fps, clip_frame_count,
            segment_start, segment_end, scene_env, scene_type, weather, lighting, time_of_day,
            person_count, driving_context, road_user_density, traffic_flow, semantic_summary,
            embedding_model, annotation_model, processing_ts, quality_flag, ego_avg_speed_mps,
            ego_max_speed_mps
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """

        formatted_data = []
        for item in batch_data:
            row = _normalize_record(item)
            formatted_data.append(
                (
                    row["video_file_name"],
                    row["video_file_path"],
                    row["dataset_name"],
                    row["dataset_version"],
                    row["dataset_split"],
                    row["clip_id"],
                    row["segment_id"],
                    row["scene_token"],
                    row["sample_start_token"],
                    row["sample_end_token"],
                    row["camera_channel"],
                    row["log_token"],
                    row["location"],
                    row["map_token"],
                    row["clip_start_ts"],
                    row["clip_end_ts"],
                    row["clip_fps"],
                    row["clip_frame_count"],
                    row["segment_start"],
                    row["segment_end"],
                    row["scene_env"],
                    row["scene_type"],
                    row["weather"],
                    row["lighting"],
                    row["time_of_day"],
                    row["person_count"],
                    row["driving_context"],
                    row["road_user_density"],
                    row["traffic_flow"],
                    row["semantic_summary"],
                    row["embedding_model"],
                    row["annotation_model"],
                    row["processing_ts"],
                    row["quality_flag"],
                    row["ego_avg_speed_mps"],
                    row["ego_max_speed_mps"],
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


def _build_ranked_query(annotation_conditions: Dict, metadata_filters: Dict, limit: int):
    annotation_conditions = annotation_conditions or {}
    metadata_filters = metadata_filters or {}

    ann_score_cases = []
    ann_score_params = []
    ann_or_clauses = []
    ann_where_params = []

    for key, value in annotation_conditions.items():
        if key not in ANNOTATION_FIELDS:
            continue
        ann_score_cases.append(f"CASE WHEN {key} = ? THEN 1 ELSE 0 END")
        ann_score_params.append(value)
        ann_or_clauses.append(f"{key} = ?")
        ann_where_params.append(value)

    meta_and_clauses = []
    meta_params = []
    for key, value in metadata_filters.items():
        if key not in METADATA_FILTER_FIELDS:
            continue
        meta_and_clauses.append(f"{key} = ?")
        meta_params.append(value)

    if ann_score_cases:
        score_expr = " + ".join(ann_score_cases)
    else:
        score_expr = "1"

    where_parts = []
    params = []

    if ann_or_clauses:
        where_parts.append("(" + " OR ".join(ann_or_clauses) + ")")
        params.extend(ann_where_params)

    if meta_and_clauses:
        where_parts.append("(" + " AND ".join(meta_and_clauses) + ")")
        params.extend(meta_params)

    where_expr = " AND ".join(where_parts) if where_parts else "1=1"

    query_sql = f"""
        SELECT
            video_file_name, video_file_path, segment_start, segment_end,
            dataset_name, dataset_version, dataset_split,
            clip_id, segment_id, scene_token, sample_start_token, sample_end_token,
            camera_channel, log_token, location, map_token,
            scene_env, scene_type, weather, lighting, time_of_day, person_count,
            driving_context, road_user_density, traffic_flow, semantic_summary,
            embedding_model, annotation_model, quality_flag,
            ({score_expr}) AS match_count
        FROM video_clips
        WHERE {where_expr}
        ORDER BY match_count DESC
        LIMIT ?
    """

    final_params = ann_score_params + params + [limit]
    n_conditions = len([k for k in annotation_conditions if k in ANNOTATION_FIELDS])
    return query_sql, final_params, n_conditions


def query_annotation_by_conditions(
    annotation_conditions: Dict, metadata_filters: Dict, limit: int = 5
) -> List[Dict]:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query_sql, params, n_conditions = _build_ranked_query(
            annotation_conditions, metadata_filters, limit
        )
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
                    "dataset_name": row["dataset_name"],
                    "dataset_version": row["dataset_version"],
                    "dataset_split": row["dataset_split"],
                    "clip_id": row["clip_id"],
                    "segment_id": row["segment_id"],
                    "scene_token": row["scene_token"],
                    "sample_start_token": row["sample_start_token"],
                    "sample_end_token": row["sample_end_token"],
                    "camera_channel": row["camera_channel"],
                    "log_token": row["log_token"],
                    "location": row["location"],
                    "map_token": row["map_token"],
                    "scene_env": row["scene_env"],
                    "scene_type": row["scene_type"],
                    "weather": row["weather"],
                    "lighting": row["lighting"],
                    "time_of_day": row["time_of_day"],
                    "person_count": row["person_count"],
                    "driving_context": row["driving_context"],
                    "road_user_density": row["road_user_density"],
                    "traffic_flow": row["traffic_flow"],
                    "semantic_summary": row["semantic_summary"],
                    "embedding_model": row["embedding_model"],
                    "annotation_model": row["annotation_model"],
                    "quality_flag": row["quality_flag"],
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


def search_sql(
    annotation_conditions: Dict = None,
    metadata_filters: Dict = None,
    limit: int = 5,
) -> List[Dict]:
    try:
        sql_hits = query_annotation_by_conditions(
            annotation_conditions=annotation_conditions or {},
            metadata_filters=metadata_filters or {},
            limit=limit,
        )
        n = len([k for k in (annotation_conditions or {}) if k in ANNOTATION_FIELDS])
        for hit in sql_hits:
            matched = hit.pop("_match_count", n if n > 0 else 1)
            hit.pop("_n_conditions", None)
            hit["score"] = matched / n if n > 0 else 1.0
        return sql_hits
    except Exception as e:
        print(f"SQL search error: {e}")
        return []


if __name__ == "__main__":
    init_db()
