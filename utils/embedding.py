import base64
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from utils.config import (
    CLIP_BATCH_SIZE,
    CLIP_MODEL_NAME,
    MAX_RETRIES,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    RETRY_BACKOFF_FACTOR,
)

ANNOTATION_RULES = {
    "scene_env": ["indoor", "outdoor", "semi_outdoor"],
    "scene_type": [
        "street",
        "square",
        "park",
        "field",
        "mountain",
        "forest",
        "river",
        "sea",
        "gym",
        "office",
        "classroom",
        "kitchen",
        "living_room",
        "mall",
        "parking",
        "underpass",
        "other",
    ],
    "weather": ["sunny", "cloudy", "rainy", "foggy", "snowy", "unknown"],
    "lighting": ["bright", "normal", "dim", "indoor_light"],
    "time_of_day": ["dawn", "morning", "noon", "afternoon", "dusk", "night", "unknown"],
    "person_count": ["0", "single", "few", "crowd"],
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
EMBEDDING_DIM = model.config.projection_dim  # Typically 512


def validate_video_file(video_path):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise IsADirectoryError(f"Path is not a file: {video_path}")

    valid_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    if video_path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Unsupported video format: {video_path.suffix}")
    return str(video_path)


def read_video_frames(
    video_file_path, segment_start, segment_end, sample_rate, frame_processor
):
    """
    read video frames
    :param video_file_path
    :param segment_start
    :param segment_end
    :param sample_rate
    :param frame_processor
    :return: processed frame results list
    """
    try:
        video_file_path = validate_video_file(video_file_path)
        cap = cv2.VideoCapture(video_file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError(f"Invalid FPS value for video: {video_file_path}")

        start_frame = int(segment_start * fps)
        end_frame = int(segment_end * fps)
        frame_count = 0
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if (
                start_frame <= frame_count < end_frame
                and frame_count % sample_rate == 0
            ):
                try:
                    results.append(frame_processor(frame))
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")

            frame_count += 1

        cap.release()

        if not results:
            print(
                f"No frames processed for video {video_file_path} (start={segment_start}, end={segment_end})"
            )
        return results

    except Exception as e:
        print(f"Error reading video frames: {str(e)}")
        return None


def validate_annotation_output(output):
    """Validate that LLM output strictly follows required JSON schema"""
    if not isinstance(output, dict):
        return False

    for field, allowed_values in ANNOTATION_RULES.items():
        if field not in output:
            return False

        field_value = str(output[field]).strip().lower()
        if field_value not in allowed_values:
            return False

    return True


def _get_default_annotation():
    return {
        "scene_env": "unknown",
        "scene_type": "other",
        "weather": "unknown",
        "lighting": "normal",
        "time_of_day": "unknown",
        "person_count": "0",
    }


def annotate(
    video_file_path, segment_start, segment_end, sample_rate, max_retries=MAX_RETRIES
):
    def frame_to_base64(frame):
        """Convert OpenCV frame to base64 string"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode(".jpg", frame, encode_param)
        return base64.b64encode(buffer).decode("utf-8")

    images = read_video_frames(
        video_file_path=video_file_path,
        segment_start=segment_start,
        segment_end=segment_end,
        sample_rate=sample_rate,
        frame_processor=frame_to_base64,
    )

    if not images:
        return _get_default_annotation()

    prompt = f"""
You are a professional video scene annotation assistant.
Output ONLY a valid JSON object with NO extra text.
Annotation rules:
{json.dumps(ANNOTATION_RULES, indent=2)}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": images}],
        "stream": False,
        "format": "json",
    }

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                wait_time = RETRY_BACKOFF_FACTOR ** (attempt - 1)
                print(f"Retry {attempt}/{max_retries} - waiting {wait_time:.1f}s...")
                time.sleep(wait_time)

            resp = requests.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=OLLAMA_TIMEOUT,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()

            result = resp.json()
            content = result["message"]["content"].strip()
            output = json.loads(content)

            if validate_annotation_output(output):
                return output
            else:
                print(f"Attempt {attempt+1}: Invalid annotation format - {content}")

        except json.JSONDecodeError as e:
            print(f"Attempt {attempt+1}: JSON decode error - {str(e)}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1}: Network/API error - {str(e)}")
        except Exception as e:
            print(f"Attempt {attempt+1}: Unexpected error - {str(e)}")

    return _get_default_annotation()


def ts_model(frame_embeddings):
    frame_embeddings = np.array(frame_embeddings)
    frame_embeddings = frame_embeddings / np.linalg.norm(
        frame_embeddings, axis=-1, keepdims=True
    )
    video_embedding = frame_embeddings.mean(axis=0)
    video_embedding = video_embedding / np.linalg.norm(video_embedding)

    return video_embedding.tolist()


def generate_video_embedding(
    video_file_path, segment_start, segment_end, sample_rate, dim
):
    def frame_to_pil(frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    pil_frames = read_video_frames(
        video_file_path=video_file_path,
        segment_start=segment_start,
        segment_end=segment_end,
        sample_rate=sample_rate,
        frame_processor=frame_to_pil,
    )

    if not pil_frames:
        return None

    frame_embeddings = []
    with torch.no_grad():
        for i in range(0, len(pil_frames), CLIP_BATCH_SIZE):
            batch_frames = pil_frames[i : i + CLIP_BATCH_SIZE]

            inputs = processor(images=batch_frames, return_tensors="pt").to(DEVICE)

            image_features = model.get_image_features(**inputs)

            normalized = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            frame_embeddings.extend(normalized.squeeze().cpu().numpy())

            del inputs, image_features, normalized
            torch.cuda.empty_cache()

    return ts_model(frame_embeddings)
