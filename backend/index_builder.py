"""
FAISS IVF index builder (Inner Product metric for normalized ViT features)
"""

import faiss
import numpy as np
import json
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexBuilder:
    """Build & save FAISS IVF-Flat index."""

    def __init__(self, n_regions: int = 50, nprobe: int = 10, use_gpu: bool = False):
        self.n_regions = n_regions
        self.nprobe = nprobe
        self.use_gpu = use_gpu

    def build_index(
        self,
        features: np.ndarray,
        image_paths: List[str],
        feature_dim: int,
        index_path: str,
        metadata_path: str
    ):
        num_images = len(features)
        logger.info(f"Building IVF index for {num_images} images...")

        # Set clusters
        nlist = min(self.n_regions, max(1, num_images // 10))

        # Quantizer for IVF
        quantizer = faiss.IndexFlatIP(feature_dim)

        index = faiss.IndexIVFFlat(
            quantizer,
            feature_dim,
            nlist,
            faiss.METRIC_INNER_PRODUCT
        )

        logger.info("Training FAISS index...")
        index.train(features)

        logger.info("Adding feature vectors...")
        index.add(features)

        index.nprobe = min(self.nprobe, nlist)

        # Save index
        faiss.write_index(index, index_path)
        logger.info(f"Index saved: {index_path}")

        # Save metadata
        metadata = {
            "num_images": num_images,
            "feature_dim": feature_dim,
            "n_regions": nlist,
            "nprobe": index.nprobe,
            "image_paths": image_paths,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved: {metadata_path}")

        return metadata
