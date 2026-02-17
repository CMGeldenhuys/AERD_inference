"""
PyTorch Hub configuration for AERD.

Usage:
    >>> import torch
    >>> model, preprocess = torch.hub.load("CMGeldenhuys/AERD", "aerd", weights="DEFAULT")
    >>> audio = torch.randn(1, 16000 * 10)  # 10 seconds at 16kHz
    >>> logits = model(audio)
    >>> probs = torch.sigmoid(logits)
"""

dependencies = ["torch", "torchaudio"]

from aerd_inference.weights import AERD_Weights, aerd, aerd_ensemble

__all__ = ["aerd", "aerd_ensemble", "AERD_Weights"]
