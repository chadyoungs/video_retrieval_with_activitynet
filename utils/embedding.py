import numpy as np
import cv2
from PIL import Image

import base64
import json
import requests
import torch

from transformers import CLIPProcessor, CLIPModel

from utils.config import ALIAS, CLIP_MODEL_NAME, OLLAMA_API_URL, OLLAMA_MODEL


model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
# embedding dim for video is different from text
EMBEDDING_DIM = model.config.projection_dim  # Typically 512


def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string"""
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

def validate_annotation_output(output):
    """
    Validate that LLM output strictly follows required JSON schema
    """
    required_fields = {
        "scene_env": ["indoor", "outdoor", "semi_outdoor"],
        "scene_type": ["street", "square", "park", "field", "mountain", "forest", "sea",
                       "gym", "office", "classroom", "kitchen", "living_room", "mall",
                       "parking", "underpass", "other"],
        "weather": ["sunny", "cloudy", "rainy", "foggy", "snowy", "unknown"],
        "lighting": ["bright", "normal", "dim", "indoor_light"],
        "time_of_day": ["dawn", "morning", "noon", "afternoon", "dusk", "night", "unknown"],
        "person_count": ["0", "single", "few", "crowd"]
    }

    if not isinstance(output, dict):
        return False

    for field, allowed in required_fields.items():
        if field not in output:
            return False
        if str(output[field]) not in allowed:
            return False

    return True

def annotate(video_file_path, segment_start, segment_end, sample_rate, max_retries=3):
    """
    Annotate frames using local Ollama multimodal model
    with strict JSON output validation & retry logic
    """
    cap = cv2.VideoCapture(video_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file: {video_file_path}")
        return None

    frame_count = 0
    images = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count >= int(segment_start * fps) and frame_count < int(segment_end * fps):
            if frame_count % int(sample_rate) == 0:
                image = frame_to_base64(frame)
                images.append(image)
        
        frame_count += 1
    
    cap.release()
    
    if not images:
        print("No frames were processed for annotation.")
        return None
    
    prompt = """
You are a professional video scene annotation assistant.
Follow the rules strictly.
Output ONLY a valid JSON object. NO extra text, NO explanation, NO markdown.

Annotation rules:
1. scene_env: indoor / outdoor / semi_outdoor
2. scene_type: street / square / park / field / mountain / forest / river / sea / gym / office / classroom / kitchen / living_room / mall / parking / underpass / other
3. weather: sunny / cloudy / rainy / foggy / snowy / unknown
4. lighting: bright / normal / dim / indoor_light
5. time_of_day: dawn / morning / noon / afternoon / dusk / night / unknown
6. person_count: 0 / single / few / crowd

Return ONLY JSON.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images
            }
        ],
        "stream": False,
        "format": "json"
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(OLLAMA_API_URL, json=payload, timeout=600)
            resp.raise_for_status()
            result = resp.json()
            content = result["message"]["content"]
            output = json.loads(content)
            
            if validate_annotation_output(output):
                return output
            else:
                print(f"Attempt {attempt+1}: Invalid format, retrying...")

        except Exception as e:
            print(f"Attempt {attempt+1}: Error - {str(e)}")

    return {
        "scene_env": "unknown",
        "scene_type": "other",
        "weather": "unknown",
        "lighting": "normal",
        "time_of_day": "unknown",
        "person_count": "0"
    }

def ts_model(frame_embeddings):
    frame_embeddings = torch.from_numpy(np.array(frame_embeddings)) 
    frame_embeddings = frame_embeddings / frame_embeddings.norm(dim=-1, keepdim=True)
    video_embedding = frame_embeddings.mean(dim=0)
    video_embedding = video_embedding / video_embedding.norm()
    
    return video_embedding.cpu().numpy().tolist()

def generate_video_embedding(video_file_path, segment_start, segment_end, sample_rate, dim):
    """
    Extracts frames, generates CLIP embeddings, and aggregates them.
    """
    frame_embeddings = []
    cap = cv2.VideoCapture(video_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file: {video_file_path}")
        return None

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
       
        if frame_count >= int(segment_start * fps) and frame_count < int(segment_end * fps):
            if frame_count % int(sample_rate) == 0:
                # Convert OpenCV BGR frame to PIL RGB Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # Generate CLIP embedding
                inputs = processor(images=pil_image, return_tensors="pt")
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)

                # Append L2-normalized vector
                image_features = image_features.pooler_output
                normalized_feature = image_features / image_features.norm(
                    p=2, dim=-1, keepdim=True
                )
                frame_embeddings.append(normalized_feature.squeeze().cpu().numpy())

        frame_count += 1

    cap.release()

    if not frame_embeddings:
        print("No frames were processed.")
        return None

    return ts_model(frame_embeddings)
