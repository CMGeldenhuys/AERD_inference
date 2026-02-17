"""Shared pytest fixtures for AERD inference tests."""

import pytest
import torch

from aerd.labels import LABEL_MAPS
from aerd.model import AERDClassifier


_EV_CALL_LABELS = LABEL_MAPS[("elephant-voices", "call")]


@pytest.fixture
def ev_call_model():
    """AERDClassifier with seq predictor and EV_CALL labels."""
    return AERDClassifier(
        num_classes=5,
        predictor_type="seq",
        class_labels=_EV_CALL_LABELS,
    )


@pytest.fixture
def label_model():
    """AERDClassifier with label predictor and EV_CALL labels."""
    return AERDClassifier(
        num_classes=5,
        predictor_type="label",
        class_labels=_EV_CALL_LABELS,
    )


@pytest.fixture
def model_no_labels():
    """AERDClassifier with no class_labels."""
    return AERDClassifier(num_classes=5, class_labels=None)


@pytest.fixture
def dummy_audio():
    """Batch of 2 one-second waveforms at 16 kHz."""
    return torch.randn(2, 16000)


@pytest.fixture
def dummy_audio_1d():
    """Single unbatched one-second waveform at 16 kHz."""
    return torch.randn(16000)
