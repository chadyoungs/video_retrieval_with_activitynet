#!/usr/bin/env python3
import json
import sqlite3

DB_NAME = "./database/video_metadata.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        DROP TABLE IF EXISTS video_clips
    ''')
    c.execute('''
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
    ''')
    conn.commit()
    conn.close()


def save_to_db(video_file_name, video_file_path, segment_start, segment_end, anno):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO video_clips (
            video_file_name, video_file_path, segment_start, segment_end, scene_env, scene_type, weather, lighting,
            time_of_day, person_count
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
    ''', (
        video_file_name,
        video_file_path,
        segment_start,
        segment_end,
        anno.get("scene_env"),
        anno.get("scene_type"),
        anno.get("weather"),
        anno.get("lighting"),
        anno.get("time_of_day"),
        anno.get("person_count")
    ))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()