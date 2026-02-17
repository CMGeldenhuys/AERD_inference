"""
AERD - Automated Elephant Rumble Detection.

A deep learning model for detecting and classifying elephant vocalizations
using the BEATs audio foundation model.
"""

from .model import AERDClassifier
from .weights import AERD_Weights, aerd, aerd_ensemble
from .labels import LABEL_MAPS, get_class_labels
from .beats import (
    BEATs,
    BEATsConfig,
    build_beats_model,
    beats_preprocessing,
    beats_extract_features,
)

import importlib.metadata

try:
    __version__ = importlib.metadata.version("aerd_inference")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode

__all__ = [
    # Main API
    "AERDClassifier",
    "AERD_Weights",
    "aerd",
    "aerd_ensemble",
    # Labels
    "LABEL_MAPS",
    "get_class_labels",
    # BEATs utilities
    "BEATs",
    "BEATsConfig",
    "build_beats_model",
    "beats_preprocessing",
    "beats_extract_features",
]
