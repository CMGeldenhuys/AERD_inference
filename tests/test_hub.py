"""Smoke tests for torch hub entry points."""

from unittest.mock import patch

import torch

import hubconf
from aerd.model import AERDClassifier
from aerd.weights import aerd, aerd_ensemble


def test_aerd_returns_model_and_preprocess():
    model, preprocess = aerd(weights=None)
    assert isinstance(model, AERDClassifier)
    assert callable(preprocess)


def test_aerd_forward_pass():
    model, _preprocess = aerd(weights=None)
    audio = torch.randn(1, 16000 * 10)
    out = model(audio)
    assert isinstance(out, torch.Tensor)


def test_aerd_ensemble_returns_list_and_preprocess():
    # Patch _load_weights to return a fake checkpoint so no download occurs
    fake_ckpt = {
        "hyper_parameters": {"num_classes": 5, "predictor": "seq"},
        "state_dict": AERDClassifier(num_classes=5).state_dict(),
    }
    with patch("aerd.weights._load_weights", return_value=fake_ckpt):
        models, preprocess = aerd_ensemble(folds=[1, 2])
    assert isinstance(models, list) and len(models) == 2
    for m in models:
        assert isinstance(m, AERDClassifier)
    assert callable(preprocess)


def test_hubconf_exports():
    assert hasattr(hubconf, "__all__")
    for name in ("aerd", "aerd_ensemble", "AERD_Weights"):
        assert name in hubconf.__all__
