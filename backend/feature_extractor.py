"""
Feature extraction using TorchVision ViT-B/16
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
from typing import Optional, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract 768-dim ViT-B/16 image embeddings (CLS token)."""

    def __init__(self, device: Optional[str] = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        logger.info("Loading ViT-B/16 (TorchVision) model...")
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.model = vit_b_16(weights=weights)

        # Modify model forward → return CLS token embedding
        self.model.forward = self._forward_features
        self.model.to(self.device).eval()

        self.feature_dim = 768

        # ViT transforms
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def _forward_features(self, x):
        x = self.model._process_input(x)
        n = x.shape[0]
        cls = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.model.encoder(x)
        return x[:, 0]  # CLS token

    

    @torch.no_grad()
    def extract_features(self, image_path: str) -> np.ndarray:
        """Return a normalized 768-D embedding."""
        try:
            img = Image.open(image_path).convert("RGB")
            img = self.transform(img).unsqueeze(0).to(self.device)

            feat = self.model(img).cpu().numpy().squeeze()

            # L2 normalize
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm

            return feat

        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return None

    def extract_from_directory(
        self, 
        directory_path: str, 
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        """
        Extract features from all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            progress_callback: Optional callback function(current, total, current_image_path)
        
        Returns:
            tuple: (features_array, image_paths_list, metadata_dict)
        """
        import glob
        
        # Supported image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
        
        # Find all image files
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(directory_path, ext), recursive=False))
            image_paths.extend(glob.glob(os.path.join(directory_path, ext.upper()), recursive=False))
            # Also search in subdirectories
            image_paths.extend(glob.glob(os.path.join(directory_path, '**', ext), recursive=True))
            image_paths.extend(glob.glob(os.path.join(directory_path, '**', ext.upper()), recursive=True))
        
        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))
        
        if len(image_paths) == 0:
            logger.warning(f"No images found in {directory_path}")
            return np.array([]), [], {}
        
        logger.info(f"Found {len(image_paths)} images. Extracting features...")
        
        features_list = []
        valid_image_paths = []
        
        for idx, img_path in enumerate(image_paths):
            try:
                feat = self.extract_features(img_path)
                if feat is not None:
                    features_list.append(feat)
                    valid_image_paths.append(img_path)
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(idx + 1, len(image_paths), img_path)
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        if len(features_list) == 0:
            logger.warning("No valid features extracted")
            return np.array([]), [], {}
        
        # Convert to numpy array
        features_array = np.array(features_list).astype('float32')
        
        # Create metadata
        metadata = {
            'num_images': len(valid_image_paths),
            'feature_dim': self.feature_dim,
            'device': str(self.device)
        }
        
        logger.info(f"Successfully extracted features from {len(valid_image_paths)} images")
        
        return features_array, valid_image_paths, metadata
