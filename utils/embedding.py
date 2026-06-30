import base64
import json
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import cv2
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from utils.config import (
    ANNOTATION_MODEL_VERSION,
    CLIP_MODEL_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_VERSION,
    MAX_RETRIES,
    N_VLM_FRAMES,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    RETRY_BACKOFF_FACTOR,
)
from utils.keyframeselection import select_frames

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
    "time_of_day": [
        "dawn",
        "morning",
        "noon",
        "afternoon",
        "dusk",
        "night",
        "unknown",
    ],
    "person_count": ["0", "single", "few", "crowd"],
    "driving_context": [
        "city_driving",
        "highway",
        "intersection",
        "parking_lot",
        "residential",
        "unknown",
    ],
    "road_user_density": ["empty", "sparse", "moderate", "dense", "unknown"],
    "traffic_flow": ["free_flow", "slow", "congested", "stopped", "unknown"],
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_processor = None


def get_processor():
    global _processor
    if _processor is None:
        _processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
    return _processor


def get_model():
    global _model
    if _model is None:
        _model = AutoModel.from_pretrained(CLIP_MODEL_NAME).eval()
    return _model


class _LazyModel:
    def __getattr__(self, name):
        return getattr(get_model(), name)

    def __call__(self, *args, **kwargs):
        return get_model()(*args, **kwargs)


class _LazyProcessor:
    def __getattr__(self, name):
        return getattr(get_processor(), name)

    def __call__(self, *args, **kwargs):
        return get_processor()(*args, **kwargs)


model = _LazyModel()
processor = _LazyProcessor()


def _extract_model_features(output):
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state.mean(dim=1)
    if isinstance(output, tuple) and len(output) > 0:
        out0 = output[0]
        if out0.ndim == 3:
            return out0.mean(dim=1)
        return out0
    raise ValueError("Unable to extract features from model output")


def get_text_features(model_obj, text_inputs):
    if hasattr(model_obj, "get_text_features"):
        features = model_obj.get_text_features(**text_inputs)
    else:
        features = _extract_model_features(model_obj(**text_inputs))
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features


def get_image_features(model_obj, image_inputs):
    if hasattr(model_obj, "get_image_features"):
        features = model_obj.get_image_features(**image_inputs)
    else:
        features = _extract_model_features(model_obj(**image_inputs))
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features


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


def read_video_frames_raw(video_file_path, segment_start, segment_end, sample_rate):
    try:
        video_file_path = validate_video_file(video_file_path)
        cap = cv2.VideoCapture(video_file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError(f"Invalid FPS value for video: {video_file_path}")

        start_frame = int(segment_start * fps)
        end_frame = int(segment_end * fps)
        frame_count = 0
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if (
                start_frame <= frame_count < end_frame
                and frame_count % sample_rate == 0
            ):
                frames.append(frame)

            frame_count += 1
            if frame_count >= end_frame:
                break

        cap.release()

        if not frames:
            print(
                f"No frames read for {video_file_path} "
                f"(start={segment_start}, end={segment_end})"
            )
        return frames

    except Exception as e:
        print(f"Error reading video frames: {str(e)}")
        return None


def read_video_frames(
    video_file_path, segment_start, segment_end, sample_rate, frame_processor
):
    raw_frames = read_video_frames_raw(
        video_file_path, segment_start, segment_end, sample_rate
    )
    if not raw_frames:
        return raw_frames

    results = []
    for frame in raw_frames:
        try:
            results.append(frame_processor(frame))
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
    return results


def validate_annotation_output(output):
    if not isinstance(output, dict):
        return False

    required_fields = list(ANNOTATION_RULES.keys()) + ["semantic_summary"]
    for field in required_fields:
        if field not in output:
            return False

    for field, allowed_values in ANNOTATION_RULES.items():
        field_value = str(output[field]).strip().lower()
        if field_value not in allowed_values:
            return False

    if not isinstance(output.get("semantic_summary"), str):
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
        "driving_context": "unknown",
        "road_user_density": "unknown",
        "traffic_flow": "unknown",
        "semantic_summary": "unknown",
    }


def annotate(
    video_file_path,
    segment_start,
    segment_end,
    sample_rate,
    max_retries=MAX_RETRIES,
    preloaded_frames=None,
):
    def frame_to_base64(frame):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode(".jpg", frame, encode_param)
        return base64.b64encode(buffer).decode("utf-8")

    if preloaded_frames is not None:
        raw_frames = preloaded_frames
    else:
        raw_frames = read_video_frames_raw(
            video_file_path, segment_start, segment_end, sample_rate
        )

    if not raw_frames:
        return _get_default_annotation()

    keyframes = select_frames(raw_frames, N_VLM_FRAMES)
    images = [frame_to_base64(f) for f in keyframes]

    prompt = (
        """
        You are a strict video scene annotation expert.
        YOU MUST FOLLOW ALL RULES 100% STRICTLY.
        YOU ONLY OUTPUT A SINGLE JSON OBJECT, NO OTHER TEXT, NO EXPLANATION, NO MARKDOWN, NO CHATTING.

        ### RULES (YOU CANNOT CHOOSE VALUES OUTSIDE THESE LISTS)
        """
        + json.dumps(ANNOTATION_RULES, indent=2)
        + """

        ### EXTRA RULES
        - semantic_summary must be a concise single-sentence summary (max 30 words).

        ### OUTPUT FORMAT (ONLY JSON, NO OTHER CONTENT)
        {
            "scene_env": "[value only from list]",
            "scene_type": "[value only from list]",
            "weather": "[value only from list]",
            "lighting": "[value only from list]",
            "time_of_day": "[value only from list]",
            "person_count": "[value only from list]",
            "driving_context": "[value only from list]",
            "road_user_density": "[value only from list]",
            "traffic_flow": "[value only from list]",
            "semantic_summary": "[short plain-text summary]"
        }

        NOW ANNOTATE THE PROVIDED IMAGES.
        ONLY OUTPUT JSON.
    """
    )

    json_schema = {
        "type": "object",
        "properties": {
            **{
                k: {"type": "string", "enum": v}
                for k, v in ANNOTATION_RULES.items()
            },
            "semantic_summary": {"type": "string"},
        },
        "required": list(ANNOTATION_RULES.keys()) + ["semantic_summary"],
    }
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": images}],
        "stream": False,
        "format": json_schema,
        "options": {"think": False},
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
            message = result.get("message", {})
            content = message.get("content", "").strip()
            if not content:
                print(f"Attempt {attempt+1}: Empty content in response")
                continue
            output = json.loads(content)

            if validate_annotation_output(output):
                print(
                    f"Annotation successful on attempt {attempt+1} for "
                    f"{os.path.basename(video_file_path)} "
                    f"[{segment_start:.1f}s - {segment_end:.1f}s]"
                )
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
    video_file_path,
    segment_start,
    segment_end,
    sample_rate,
    dim,
    preloaded_frames=None,
):
    def frame_to_pil(frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    if preloaded_frames is not None:
        pil_frames = [frame_to_pil(f) for f in preloaded_frames]
    else:
        pil_frames = read_video_frames(
            video_file_path=video_file_path,
            segment_start=segment_start,
            segment_end=segment_end,
            sample_rate=sample_rate,
            frame_processor=frame_to_pil,
        )

    if not pil_frames:
        return None

    _model = get_model()
    _processor = get_processor()

    frame_embeddings = []
    with torch.no_grad():
        batch_frames = select_frames(pil_frames, N_VLM_FRAMES)
        inputs = _processor(images=batch_frames, return_tensors="pt")
        image_features = get_image_features(_model, inputs)

        normalized = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        frame_embeddings.extend(normalized.squeeze().cpu().numpy())

        del inputs, image_features, normalized

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"Generated {len(frame_embeddings)} frame embeddings ({EMBEDDING_MODEL_VERSION}) for "
        f"{os.path.basename(video_file_path)} "
        f"[{segment_start:.1f}s - {segment_end:.1f}s]"
    )
    return ts_model(frame_embeddings)
