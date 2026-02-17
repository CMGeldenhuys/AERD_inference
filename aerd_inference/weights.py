"""
AERD Model Weights Management.

Provides utilities for loading pretrained AERD model weights from GitHub Releases.
Supports per-fold and ensemble loading for k-fold cross-validation models.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Tuple

import torch
from torch.hub import load_state_dict_from_url

from .labels import LABEL_MAPS

if TYPE_CHECKING:
    from .model import AERDClassifier


# GitHub Release URL template
# Update this URL when publishing new releases
_WEIGHTS_URL_BASE = "https://github.com/CMGeldenhuys/AERD/releases/download"


class AERD_Weights(Enum):
    """
    Available pretrained weights for AERD models.

    Each weight variant contains:
    - url_template: URL template with {version} and {fold} placeholders
    - version: Default version tag
    - num_classes: Number of output classes
    - num_folds: Number of k-fold splits
    - class_labels: Mapping from class index to label name
    - meta: Additional metadata (dataset, variant, etc.)
    """

    EV_CALL = {
        "url_template": f"{_WEIGHTS_URL_BASE}/v{{version}}/aerd_ev_call_ko{{fold}}_v{{version}}.pt",
        "version": "1",
        "num_classes": 5,
        "num_folds": 5,
        "class_labels": LABEL_MAPS[("elephant-voices", "call")],
        "meta": {
            "dataset": "elephant-voices",
            "variant": "call",
            "description": "AERD model for elephant call-type classification (EV dataset)",
        },
    }

    EV_SUBCALL = {
        "url_template": f"{_WEIGHTS_URL_BASE}/v{{version}}/aerd_ev_subcall_ko{{fold}}_v{{version}}.pt",
        "version": "1",
        "num_classes": 8,
        "num_folds": 5,
        "class_labels": LABEL_MAPS[("elephant-voices", "subcall")],
        "meta": {
            "dataset": "elephant-voices",
            "variant": "subcall",
            "description": "AERD model for elephant sub-call classification (EV dataset)",
        },
    }

    EV_BINARY = {
        "url_template": f"{_WEIGHTS_URL_BASE}/v{{version}}/aerd_ev_binary_ko{{fold}}_v{{version}}.pt",
        "version": "1",
        "num_classes": 2,
        "num_folds": 5,
        "class_labels": LABEL_MAPS[("elephant-voices", "binary")],
        "meta": {
            "dataset": "elephant-voices",
            "variant": "binary",
            "description": "AERD binary call detection model (EV dataset)",
        },
    }

    LDC_POOLE = {
        "url_template": f"{_WEIGHTS_URL_BASE}/v{{version}}/aerd_ldc_poole_ko{{fold}}_v{{version}}.pt",
        "version": "1",
        "num_classes": 6,
        "num_folds": 10,
        "class_labels": LABEL_MAPS[("asian-voc", "poole")],
        "meta": {
            "dataset": "asian-voc",
            "variant": "poole",
            "description": "AERD model for Asian elephant call classification (LDC dataset, Poole taxonomy)",
        },
    }

    LDC_BINARY = {
        "url_template": f"{_WEIGHTS_URL_BASE}/v{{version}}/aerd_ldc_binary_ko{{fold}}_v{{version}}.pt",
        "version": "1",
        "num_classes": 2,
        "num_folds": 10,
        "class_labels": LABEL_MAPS[("asian-voc", "binary")],
        "meta": {
            "dataset": "asian-voc",
            "variant": "binary",
            "description": "AERD binary call detection model (LDC dataset)",
        },
    }

    DEFAULT = EV_CALL

    @property
    def url_template(self) -> str:
        return self.value["url_template"]

    @property
    def version(self) -> str:
        return self.value["version"]

    @property
    def num_classes(self) -> int:
        return self.value["num_classes"]

    @property
    def num_folds(self) -> int:
        return self.value["num_folds"]

    @property
    def class_labels(self) -> dict[int, str]:
        return self.value["class_labels"]

    @property
    def meta(self) -> dict:
        return self.value["meta"]

    def get_url(self, fold: int, version: str | None = None) -> str:
        """Build the download URL for a specific fold and version."""
        v = version or self.version
        return self.url_template.format(version=v, fold=fold)


def _load_weights(
    weights: AERD_Weights | str | None,
    fold: int = 1,
    progress: bool = True,
    check_hash: bool = False,
) -> dict | None:
    """
    Load weights from URL or local path.

    Args:
        weights: Weights specification (AERD_Weights enum, URL string, or local path).
        fold: Fold number (only used with AERD_Weights enum).
        progress: Show download progress bar.
        check_hash: Verify file hash after download.

    Returns:
        Loaded checkpoint dictionary, or None if weights is None.
    """
    if weights is None:
        return None

    if isinstance(weights, AERD_Weights):
        url = weights.get_url(fold)
        return load_state_dict_from_url(
            url,
            progress=progress,
            check_hash=check_hash,
            map_location="cpu",
            weights_only=True,
        )
    elif isinstance(weights, str):
        if weights.startswith(("http://", "https://")):
            return load_state_dict_from_url(
                weights,
                progress=progress,
                check_hash=check_hash,
                map_location="cpu",
                weights_only=True,
            )
        else:
            # Local file path
            return torch.load(weights, map_location="cpu", weights_only=True)
    else:
        raise TypeError(f"weights must be AERD_Weights, str, or None, got {type(weights)}")


def _build_model_from_checkpoint(
    checkpoint: dict,
    weights: AERD_Weights | str | None = None,
) -> "AERDClassifier":
    """Build an AERDClassifier from a loaded checkpoint dict."""
    from .model import AERDClassifier

    hparams = checkpoint.get("hyper_parameters", {})
    beats_config = checkpoint.get("beats_config", None)
    class_labels = checkpoint.get("class_labels", None)
    dataset_info = checkpoint.get("dataset_info", None)

    # Get metadata from weights enum if available
    if isinstance(weights, AERD_Weights):
        meta = weights.meta
        num_classes = weights.num_classes
        if class_labels is None:
            class_labels = weights.class_labels
    else:
        meta = {}
        num_classes = hparams.get("num_classes", 5)

    model = AERDClassifier(
        num_classes=num_classes,
        fbank_mean=hparams.get("fbank_mean", meta.get("fbank_mean", 14.7274)),
        fbank_std=hparams.get("fbank_std", meta.get("fbank_std", 2.9366)),
        predictor_type=hparams.get("predictor", meta.get("predictor", "seq")),
        beats_config=beats_config,
        class_labels=class_labels,
        dataset_info=dataset_info,
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.source_path = checkpoint.get("source_path", None)
    model.eval()
    return model


def aerd(
    weights: AERD_Weights | str = AERD_Weights.DEFAULT,
    fold: int = 1,
    progress: bool = True,
) -> Tuple["AERDClassifier", Callable]:
    """
    Load AERD model with pretrained weights.

    This is the main entry point for PyTorch Hub.

    Args:
        weights: Pretrained weights to load. Can be:
            - AERD_Weights.DEFAULT: Default pretrained weights
            - A URL string to download weights from
            - A local file path to load weights from
        fold: Fold number to load (1-indexed). Only used with AERD_Weights enum.
        progress: Show download progress bar.

    Returns:
        model: Loaded AERDClassifier model in eval mode.
        preprocess: Preprocessing function for audio input.

    Example:
        >>> import torch
        >>> model, preprocess = torch.hub.load("CMGeldenhuys/AERD", "aerd")
        >>> audio = torch.randn(1, 16000 * 10)  # 10 seconds at 16kHz
        >>> logits = model(audio)
    """
    from .model import AERDClassifier

    checkpoint = _load_weights(weights, fold=fold, progress=progress)

    if checkpoint is None:
        model = AERDClassifier()
        model.eval()
    else:
        model = _build_model_from_checkpoint(checkpoint, weights)

    return model, model.preprocess


def aerd_ensemble(
    weights: AERD_Weights = AERD_Weights.DEFAULT,
    folds: list[int] | None = None,
    progress: bool = True,
) -> Tuple[List["AERDClassifier"], Callable]:
    """
    Load an ensemble of AERD models across k-fold splits.

    Args:
        weights: Weight variant to load.
        folds: List of fold numbers to load (1-indexed). If None, loads all folds.
        progress: Show download progress bar.

    Returns:
        models: List of loaded AERDClassifier models in eval mode.
        preprocess: Preprocessing function (shared across all models).

    Example:
        >>> models, preprocess = aerd_ensemble(AERD_Weights.EV_CALL)
        >>> audio = torch.randn(1, 16000 * 10)
        >>> # Average predictions across folds
        >>> logits = torch.stack([m(audio) for m in models]).mean(dim=0)
    """
    if folds is None:
        folds = list(range(1, weights.num_folds + 1))

    models = []
    for fold in folds:
        checkpoint = _load_weights(weights, fold=fold, progress=progress)
        if checkpoint is not None:
            model = _build_model_from_checkpoint(checkpoint, weights)
            models.append(model)

    if not models:
        raise RuntimeError(f"No models loaded for {weights.name}")

    return models, models[0].preprocess
