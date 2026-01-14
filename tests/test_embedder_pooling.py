import math

import numpy as np
import torch

from thirawat_mapper_beta.models.embedder import SapBERTEmbedder


class _DummyTokenizer:
    def __call__(self, texts, **kwargs):  # noqa: ANN001
        batch = len(texts)
        # 3 tokens, last token padded out
        input_ids = torch.zeros((batch, 3), dtype=torch.long)
        attn = torch.tensor([[1, 1, 0]] * batch, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attn}


class _DummyModel:
    class _Cfg:
        hidden_size = 2

    config = _Cfg()

    def to(self, device):  # noqa: ANN001
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):  # noqa: ANN001
        # token0=[1,0], token1=[0,1], token2=[1,1] (but masked out)
        bsz = kwargs["input_ids"].shape[0]
        hidden = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]] * bsz,
            dtype=torch.float32,
        )

        class _Out:
            last_hidden_state = hidden

        return _Out()


def test_embedder_cls_pooling(monkeypatch):
    monkeypatch.setattr(
        "thirawat_mapper_beta.models.embedder.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _DummyTokenizer(),
    )
    monkeypatch.setattr(
        "thirawat_mapper_beta.models.embedder.AutoModel.from_pretrained",
        lambda *args, **kwargs: _DummyModel(),
    )

    emb = SapBERTEmbedder(device="cpu", batch_size=8, max_length=8, pooling="cls")
    vec = emb.encode(["x"])
    assert vec.shape == (1, 2)
    assert vec.dtype == np.float32
    assert np.allclose(vec[0], np.array([1.0, 0.0], dtype=np.float32), atol=1e-6)


def test_embedder_mean_pooling_respects_attention_mask(monkeypatch):
    monkeypatch.setattr(
        "thirawat_mapper_beta.models.embedder.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _DummyTokenizer(),
    )
    monkeypatch.setattr(
        "thirawat_mapper_beta.models.embedder.AutoModel.from_pretrained",
        lambda *args, **kwargs: _DummyModel(),
    )

    emb = SapBERTEmbedder(device="cpu", batch_size=8, max_length=8, pooling="mean")
    vec = emb.encode(["x"])
    assert vec.shape == (1, 2)
    # mean([1,0],[0,1]) -> [0.5,0.5] then L2-normalized
    expected = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)], dtype=np.float32)
    assert np.allclose(vec[0], expected, atol=1e-6)

