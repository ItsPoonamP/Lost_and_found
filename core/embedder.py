"""
embedder.py — CLIP-based visual embedding.

encode_image(image) → normalized float32 numpy array (512-dim for ViT-B-16)

Embeddings are L2-normalized so cosine similarity = inner product.
This means FAISS IndexFlatIP gives cosine similarity scores in [0, 1].
"""

import numpy as np
import torch
import open_clip
from PIL import Image
import config


class Embedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL,
            pretrained=config.CLIP_PRETRAINED,
        )
        self.model = self.model.to(self.device).eval()
        print(f"  CLIP {config.CLIP_MODEL} loaded on [{self.device}]")

    def encode_image(self, image) -> np.ndarray:
        """
        Args:
            image: BGR numpy array (from OpenCV) or PIL RGB Image

        Returns:
            numpy float32 array of shape (EMBEDDING_DIM,), L2-normalized
        """
        if isinstance(image, np.ndarray):
            # OpenCV BGR → PIL RGB
            image = Image.fromarray(image[..., ::-1])

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)   # normalize

        return features.cpu().numpy()[0].astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Optional: encode a text description as a query.
        e.g. embedder.encode_text("a red backpack with black straps")
        """
        tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)
        tokens = tokenizer([text]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()[0].astype(np.float32)
