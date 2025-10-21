import numpy as np
import cv2
from PIL import Image

import torch

from transformers import CLIPProcessor, CLIPModel

from config import CLIP_MODEL_NAME


model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
# embedding dim for video is different from text
EMBEDDING_DIM = model.config.projection_dim  # Typically 512


def generate_video_embedding(video_path, sample_rate, dim):
    """
    Extracts frames, generates CLIP embeddings, and aggregates them.
    """
    frame_embeddings = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Sample the frame based on rate (e.g., every 1*FPS frames for 1 second sampling)
        if frame_count % int(fps * sample_rate) == 0:
            # Convert OpenCV BGR frame to PIL RGB Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Generate CLIP embedding
            inputs = processor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)

            # Append L2-normalized vector
            normalized_feature = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )
            frame_embeddings.append(normalized_feature.squeeze().cpu().numpy())

        frame_count += 1

    cap.release()

    if not frame_embeddings:
        print("No frames were processed.")
        return None

    # AGGREGATION STRATEGY: Mean Pooling
    # The simplest and most common method to get a single video vector
    # video_vector = np.mean(frame_embeddings, axis=0)

    # Max Pooling along the first axis (axis=0), which represents the frames.
    # The result is a 1D array where each element is the maximum value
    # observed at that feature dimension across all frames.
    feature_matrix = np.array(frame_embeddings)
    video_vector = np.max(feature_matrix, axis=0)

    return video_vector.tolist()  # Convert back to list for Milvus insertion
