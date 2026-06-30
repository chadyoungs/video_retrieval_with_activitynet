"""VLM-based reranker for video retrieval results.

After an initial vector + metadata retrieval pass returns a candidate pool
(typically top-10), this module sends each candidate's keyframes to the local
VLM (Ollama) with a carefully crafted prompt and obtains a relevance score
(0–10).  Candidates are then sorted by that score and the top-k are returned.

Supported query modes
---------------------
* **Text query** – score how well a video segment matches a natural-language
  description.
* **Image query** – score how visually similar a video segment is to a
  reference image.
"""

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import requests
from PIL import Image

from utils.config import (
    MAX_RETRIES,
    N_VLM_FRAMES,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    RETRY_BACKOFF_FACTOR,
)
from utils.embedding import read_video_frames_raw
from utils.keyframeselection import select_frames


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_text_rerank_prompt(query_text: str) -> str:
    """Return a VLM prompt that scores a video segment against *query_text*.

    The prompt instructs the model to output a single JSON object with an
    integer ``score`` (0–10) and a one-sentence ``reason``.  Strict output
    constraints minimise hallucination and make parsing reliable.
    """
    return (
        "You are a video relevance scoring expert.\n\n"
        "You will be shown keyframes extracted from a video segment.\n"
        "Your task is to score how well this video segment matches the following text query.\n\n"
        f'Text query: "{query_text}"\n\n'
        "### SCORING GUIDE\n"
        "- 9-10 : Excellent match – the segment clearly depicts the described scene or activity.\n"
        "- 7-8  : Good match – most key elements from the query are visible.\n"
        "- 5-6  : Partial match – some elements match but important details differ or are absent.\n"
        "- 3-4  : Poor match – only superficial or incidental overlap with the query.\n"
        "- 1-2  : Barely relevant – the segment almost entirely misses the query intent.\n"
        "- 0    : No match – the segment is completely unrelated to the query.\n\n"
        "### RULES\n"
        "YOU MUST OUTPUT ONLY A SINGLE JSON OBJECT.\n"
        "NO EXPLANATION, NO MARKDOWN, NO EXTRA TEXT.\n\n"
        "### OUTPUT FORMAT\n"
        '{"score": <integer 0-10>, "reason": "<one sentence>"}\n\n'
        "SCORE THE PROVIDED KEYFRAMES NOW."
    )


def build_image_rerank_prompt() -> str:
    """Return a VLM prompt that scores a video segment against a reference image.

    The first image in the message is the query reference image; all subsequent
    images are keyframes from the candidate segment.  The model outputs a JSON
    object with ``score`` (0–10) and ``reason``.
    """
    return (
        "You are a video relevance scoring expert.\n\n"
        "You will be shown a set of images:\n"
        "  • IMAGE 1 is the REFERENCE QUERY IMAGE.\n"
        "  • IMAGES 2+ are KEYFRAMES from a candidate video segment.\n\n"
        "Your task is to score how visually similar the video segment is to the reference image.\n\n"
        "### SCORING GUIDE\n"
        "- 9-10 : Excellent match – same scene, objects, activity, and visual context.\n"
        "- 7-8  : Good match – most key visual elements are shared.\n"
        "- 5-6  : Partial match – shares some visual elements but differs meaningfully.\n"
        "- 3-4  : Poor match – only minor visual similarities.\n"
        "- 1-2  : Barely similar – almost no visual resemblance.\n"
        "- 0    : No visual similarity at all.\n\n"
        "### RULES\n"
        "YOU MUST OUTPUT ONLY A SINGLE JSON OBJECT.\n"
        "NO EXPLANATION, NO MARKDOWN, NO EXTRA TEXT.\n\n"
        "### OUTPUT FORMAT\n"
        '{"score": <integer 0-10>, "reason": "<one sentence>"}\n\n'
        "SCORE THE PROVIDED IMAGES NOW."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _frame_to_base64(frame) -> str:
    """Encode an OpenCV BGR frame as a base-64 JPEG string."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    _, buffer = cv2.imencode(".jpg", frame, encode_param)
    return base64.b64encode(buffer).decode("utf-8")


_RERANK_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
}


def _call_vlm_for_score(
    images_b64: List[str],
    prompt: str,
    max_retries: int = MAX_RETRIES,
) -> Optional[Dict]:
    """Call the Ollama VLM and return a parsed ``{"score": int, "reason": str}`` dict.

    Returns ``None`` if all attempts fail or return invalid output.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": images_b64}],
        "stream": False,
        "format": _RERANK_JSON_SCHEMA,
        "options": {"think": False},
    }

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                wait_time = RETRY_BACKOFF_FACTOR ** (attempt - 1)
                print(f"    Rerank retry {attempt}/{max_retries} – waiting {wait_time:.1f}s...")
                time.sleep(wait_time)

            resp = requests.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=OLLAMA_TIMEOUT,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()

            content = resp.json().get("message", {}).get("content", "").strip()
            if not content:
                print(f"    Rerank attempt {attempt+1}: empty VLM response")
                continue

            output = json.loads(content)
            score = output.get("score")
            if isinstance(score, int) and 0 <= score <= 10:
                return output

            print(f"    Rerank attempt {attempt+1}: invalid score value – {output}")

        except json.JSONDecodeError as e:
            print(f"    Rerank attempt {attempt+1}: JSON decode error – {e}")
        except requests.exceptions.RequestException as e:
            print(f"    Rerank attempt {attempt+1}: network/API error – {e}")
        except Exception as e:
            print(f"    Rerank attempt {attempt+1}: unexpected error – {e}")

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank_results(
    candidates: List[Dict],
    query_text: Optional[str] = None,
    query_image_path: Optional[str] = None,
    top_k: int = 5,
    sample_rate: int = 10,
) -> List[Dict]:
    """Rerank *candidates* using a local VLM and return the top-*k* results.

    Parameters
    ----------
    candidates:
        List of result dicts from the initial hybrid retrieval, each containing
        at least ``video_file_path``, ``video_file_name``, ``segment_start``,
        and ``segment_end``.
    query_text:
        Natural-language query string (used when reranking by text).
    query_image_path:
        Path to a reference image file (used when reranking by image).
    top_k:
        Number of results to return after reranking.
    sample_rate:
        Frame sampling rate (every N-th frame) when reading video segments.

    Returns
    -------
    List of result dicts sorted by ``rerank_score`` (descending), truncated to
    ``top_k``.  Each dict gains two extra fields: ``rerank_score`` (int, 0–10)
    and ``rerank_reason`` (str).

    Raises
    ------
    ValueError
        If neither *query_text* nor *query_image_path* is provided.
    """
    if query_text is None and query_image_path is None:
        raise ValueError("Either query_text or query_image_path must be provided")

    # Pre-encode the query reference image once (image-query mode only).
    query_image_b64: Optional[str] = None
    if query_image_path:
        try:
            pil_img = Image.open(query_image_path).convert("RGB")
            bgr_frame = np.array(pil_img)[:, :, ::-1]  # RGB → BGR for OpenCV
            query_image_b64 = _frame_to_base64(bgr_frame)
        except Exception as e:
            raise ValueError(f"Failed to load query image '{query_image_path}': {e}") from e

    scored: List[Dict] = []

    for idx, candidate in enumerate(candidates):
        video_path = os.path.join(
            candidate.get("video_file_path", ""),
            candidate.get("video_file_name", ""),
        )
        segment_start = candidate.get("segment_start", 0)
        segment_end = candidate.get("segment_end", 0)

        print(
            f"  Reranking [{idx + 1}/{len(candidates)}]: "
            f"{candidate.get('video_file_name')} "
            f"[{segment_start:.1f}s – {segment_end:.1f}s]"
        )

        raw_frames = read_video_frames_raw(
            video_path, segment_start, segment_end, sample_rate
        )

        if not raw_frames:
            print(f"    Warning: no frames available for {video_path}; score set to 0")
            scored.append(
                {**candidate, "rerank_score": 0, "rerank_reason": "frames unavailable"}
            )
            continue

        keyframes = select_frames(raw_frames, N_VLM_FRAMES)
        candidate_images_b64 = [_frame_to_base64(f) for f in keyframes]

        if query_text is not None:
            prompt = build_text_rerank_prompt(query_text)
            images_b64 = candidate_images_b64
        else:
            prompt = build_image_rerank_prompt()
            # Prepend the reference image so the model sees it first.
            images_b64 = [query_image_b64] + candidate_images_b64

        vlm_output = _call_vlm_for_score(images_b64, prompt)

        if vlm_output:
            rerank_score = vlm_output["score"]
            rerank_reason = vlm_output.get("reason", "")
            print(f"    Score: {rerank_score}/10 – {rerank_reason}")
        else:
            rerank_score = 0
            rerank_reason = "VLM scoring failed"
            print(f"    Warning: VLM scoring failed; score set to 0")

        scored.append(
            {**candidate, "rerank_score": rerank_score, "rerank_reason": rerank_reason}
        )

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]
