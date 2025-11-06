"""SapBERT embedder utilities built on Hugging Face Transformers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_ID = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"


class SapBERTEmbedder:
    """CLS embedding helper using AutoModel / AutoTokenizer."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str | None = None,
        batch_size: int = 256,
        max_length: int = 128,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModel | None = None

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self._tokenizer

    @property
    def model(self) -> AutoModel:
        if self._model is None:
            model = AutoModel.from_pretrained(self.model_id)
            model.to(self.device)
            model.eval()
            self._model = model
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalised CLS embeddings as float32."""

        if not texts:
            hidden = self.model.config.hidden_size
            return np.zeros((0, hidden), dtype=np.float32)

        outputs: list[np.ndarray] = []
        tokenizer = self.tokenizer
        model = self.model

        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                hidden_states = model(**encoded).last_hidden_state  # (B, T, H)
                cls_embeddings = hidden_states[:, 0, :]
                cls_embeddings = torch.nn.functional.normalize(cls_embeddings, dim=1)
                outputs.append(cls_embeddings.cpu().numpy().astype(np.float32, copy=False))

        return np.vstack(outputs)
