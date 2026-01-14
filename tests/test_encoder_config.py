import argparse

from thirawat_mapper_beta.infer.bulk import _resolve_encoder_config
from thirawat_mapper_beta.models.embedder import DEFAULT_MODEL_ID


def test_resolve_encoder_config_prefers_manifest_defaults():
    args = argparse.Namespace(
        encoder_model_id=None,
        encoder_pooling=None,
        encoder_max_length=None,
        encoder_trust_remote_code=None,
    )
    manifest = {"model_id": "m", "pooling": "mean", "max_length": 256, "trust_remote_code": True}
    cfg = _resolve_encoder_config(args, manifest)
    assert cfg == {"model_id": "m", "pooling": "mean", "max_length": 256, "trust_remote_code": True}


def test_resolve_encoder_config_allows_cli_overrides():
    args = argparse.Namespace(
        encoder_model_id="x",
        encoder_pooling="cls",
        encoder_max_length=64,
        encoder_trust_remote_code=False,
    )
    manifest = {"model_id": "m", "pooling": "mean", "max_length": 256, "trust_remote_code": True}
    cfg = _resolve_encoder_config(args, manifest)
    assert cfg == {"model_id": "x", "pooling": "cls", "max_length": 64, "trust_remote_code": False}


def test_resolve_encoder_config_falls_back_when_no_manifest():
    args = argparse.Namespace(
        encoder_model_id=None,
        encoder_pooling=None,
        encoder_max_length=None,
        encoder_trust_remote_code=None,
    )
    cfg = _resolve_encoder_config(args, None)
    assert cfg["model_id"] == DEFAULT_MODEL_ID
    assert cfg["pooling"] == "cls"
    assert cfg["max_length"] == 128
    assert cfg["trust_remote_code"] is False

