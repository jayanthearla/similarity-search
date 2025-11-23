"""
Image search engine using FAISS IVF index + ViT-B/16 features
"""

import json
import faiss
import numpy as np
import logging
import os
from typing import List, Dict
from backend.feature_extractor import FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchEngine:
    """Search similar images using ViT features + FAISS."""

    def __init__(self, index_path: str, metadata_path: str):
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.image_paths = self.metadata["image_paths"]
        self.feature_dim = self.metadata["feature_dim"]

        self.extractor = FeatureExtractor()

        logger.info(f"Loaded FAISS index with {len(self.image_paths)} images.")

    def search(self, query_image: str, k: int = 10) -> List[Dict]:
        q_feat = self.extractor.extract_features(query_image)
        if q_feat is None:
            raise ValueError("Could not extract features")

        q_feat = q_feat.reshape(1, -1).astype("float32")
        k = min(k, len(self.image_paths))

        distances, indices = self.index.search(q_feat, k)

        results = []
        for rank, (d, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:
                continue

            sim = float(d)  # For IP-metric, higher is better
            img_path = self.image_paths[idx]

            results.append({
                "rank": rank + 1,
                "image_path": img_path,
                "filename": os.path.basename(img_path),
                "similarity": sim,
                "distance": float(d),
            })

        return results
