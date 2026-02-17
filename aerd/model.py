"""
AERD Classifier - Automated Elephant Rumble Detection.

A clean inference model wrapping BEATs backbone with classification head.
"""

from __future__ import annotations

from math import floor
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .beats import BEATs, BEATsConfig, build_beats_model
from .utils import PatchwisePredictor


# Default BEATs config (from BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt1)
_DEFAULT_BEATS_CONFIG = {
    "input_patch_size": 16,
    "embed_dim": 512,
    "conv_bias": False,
    "encoder_layers": 12,
    "encoder_embed_dim": 768,
    "encoder_ffn_embed_dim": 3072,
    "encoder_attention_heads": 12,
    "activation_fn": "gelu",
    "layer_wise_gradient_decay_ratio": 0.6,
    "layer_norm_first": False,
    "deep_norm": True,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "activation_dropout": 0.0,
    "encoder_layerdrop": 0.0,
    "dropout_input": 0.0,
    "conv_pos": 128,
    "conv_pos_groups": 16,
    "relative_position_embedding": True,
    "num_buckets": 320,
    "max_distance": 800,
    "gru_rel_pos": True,
    "finetuned_model": True,
    "predictor_dropout": 0.0,
    "predictor_class": 527,
}


class AERDClassifier(nn.Module):
    """
    AERD Classifier for elephant call detection and classification.

    Wraps a BEATs backbone with a classification head for inference.

    Args:
        num_classes: Number of output classes.
        fbank_mean: Mean for fbank normalization.
        fbank_std: Standard deviation for fbank normalization.
        predictor_type: Type of predictor head ('seq', 'label', 'seq-spec', 'seq-full').
        beats_config: BEATs configuration dict (uses default if None).
        class_labels: Label mapping — accepts either reverse ``{int_index: str_label}``
            or forward ``{str_label: int_index}`` orientation; normalized internally.
        dataset_info: Metadata about the training dataset and fold.
    """

    SAMPLE_RATE = 16_000

    def __init__(
        self,
        num_classes: int = 5,
        fbank_mean: float = 14.7274,
        fbank_std: float = 2.9366,
        predictor_type: str = "seq",
        beats_config: dict | None = None,
        class_labels: dict[int, str] | dict[str, int] | None = None,
        dataset_info: dict | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fbank_mean = fbank_mean
        self.fbank_std = fbank_std
        self.predictor_type = predictor_type
        self.dataset_info = dataset_info

        # Normalize class_labels to {int_index: str_label} (reverse mapping)
        # and store the inverse as class_indexes {str_label: int_index} (forward mapping).
        if class_labels is not None:
            first_key = next(iter(class_labels))
            if isinstance(first_key, str):
                # Forward mapping {str_label: int_index} — invert it
                self.class_labels: dict[int, str] | None = {v: k for k, v in class_labels.items()}
                self.class_indexes: dict[str, int] | None = dict(class_labels)
            else:
                # Already reverse mapping {int_index: str_label}
                self.class_labels = dict(class_labels)
                self.class_indexes = {v: k for k, v in class_labels.items()}
        else:
            self.class_labels = None
            self.class_indexes = None

        # Build BEATs backbone with config
        if beats_config is None:
            beats_config = _DEFAULT_BEATS_CONFIG.copy()

        cfg = BEATsConfig(beats_config)
        cfg.encoder_layerdrop = 0.0  # Disable for inference
        self.backbone = BEATs(cfg)
        self.backbone.predictor = None  # We use our own predictor

        # Compute number of spectral patches
        num_mel_bins = 128
        padding = self.backbone.patch_embedding.padding
        dilation = self.backbone.patch_embedding.dilation
        stride = self.backbone.patch_embedding.stride
        kernel_size = self.backbone.patch_embedding.kernel_size
        self.num_spectral_patches = int(
            floor(
                (num_mel_bins + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
                / stride[1]
                + 1
            )
        )

        # Build predictor head
        encoder_embed_dim = cfg.encoder_embed_dim
        self.predictor = self._build_predictor(predictor_type, encoder_embed_dim, num_classes)

    def _build_predictor(
        self, predictor_type: str, embed_dim: int, num_classes: int
    ) -> nn.Module:
        """Build the classification head based on predictor type."""
        if predictor_type == "seq":
            # Mean pool mel features then project
            return PatchwisePredictor(
                nn.Linear(embed_dim, num_classes),
                proj_over="mean_spec",
            )
        elif predictor_type == "label":
            # Single label per input
            from .utils import HardLabelPredictor
            return HardLabelPredictor(nn.Linear(embed_dim, num_classes))
        elif predictor_type == "seq-spec":
            # Flatten mel dim into embedding
            return PatchwisePredictor(
                nn.Linear(embed_dim * self.num_spectral_patches, num_classes),
                proj_over="all_spec",
            )
        elif predictor_type == "seq-full":
            # Prediction for each (T, M) patch
            return PatchwisePredictor(
                nn.Linear(embed_dim, num_classes),
                proj_over=None,
            )
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")

    def get_class_label(self, index: int) -> str:
        """Look up the string label for a class index."""
        if self.class_labels is None:
            raise ValueError("class_labels not available.")
        return self.class_labels[index]

    def get_class_index(self, label: str) -> int:
        """Look up the integer index for a class label."""
        if self.class_indexes is None:
            raise ValueError("class_indexes not available.")
        return self.class_indexes[label]

    def preprocess(
        self,
        waveform: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Preprocess raw waveform to mel-spectrogram (fbank).

        Args:
            waveform: Raw audio tensor of shape (batch, samples) or (samples,).
            padding_mask: Optional boolean mask indicating padded positions.

        Returns:
            fbank: Mel-spectrogram of shape (batch, time, mel_bins).
            padding_mask: Padding mask for the fbank.
        """
        # Handle single waveform
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        batch_size, audio_len = waveform.shape

        # Compute fbank using BEATs preprocessing
        fbank = self.backbone.preprocess(
            waveform,
            fbank_mean=self.fbank_mean,
            fbank_std=self.fbank_std,
        )

        # Create or propagate padding mask
        if padding_mask is None:
            padding_mask = torch.zeros(
                batch_size, audio_len, device=waveform.device, dtype=torch.bool
            )

        padding_mask = self.backbone.forward_padding_mask(fbank, padding_mask)

        return fbank, padding_mask

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
        return_features: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Either raw waveform (batch, samples) or preprocessed fbank (batch, time, mel).
            padding_mask: Optional padding mask.
            return_features: If True, also return intermediate features.

        Returns:
            logits: Classification logits of shape (batch, time, num_classes) or (batch, num_classes).
            features: (optional) Intermediate features if return_features=True.
        """
        # Check if input is raw waveform or fbank
        if x.ndim == 2 and x.shape[-1] != 128:
            # Assume raw waveform (batch, samples)
            fbank, padding_mask = self.preprocess(x, padding_mask)
        elif x.ndim == 1:
            # Single raw waveform
            fbank, padding_mask = self.preprocess(x.unsqueeze(0), padding_mask)
        else:
            fbank = x

        # Add channel dimension for backbone
        if fbank.ndim == 3:
            fbank = fbank.unsqueeze(1)  # (batch, 1, time, mel)

        # Forward through backbone
        features, attn_mask = self.backbone.forward(fbank, padding_mask=padding_mask)

        batch_size = fbank.shape[0]

        # Reshape features: (batch, tokens, embed) -> (batch, time, spectral, embed)
        features = features.reshape(batch_size, -1, self.num_spectral_patches, 768)

        # Forward through predictor
        logits = self.predictor(features, attn_mask=attn_mask)

        if return_features:
            return logits, features
        return logits

    def predict(
        self,
        waveform: Tensor,
        output: str = "tensor",
    ) -> Tensor | Dict[str, Tensor]:
        """
        Convenience method for inference.

        Args:
            waveform: Raw audio tensor.
            output: Output format — "tensor" returns probability tensor,
                "dict" returns {label_name: probability_tensor}.

        Returns:
            Probabilities as tensor or label-keyed dict.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(waveform)
            probs = torch.sigmoid(logits)

            if output == "tensor":
                return probs
            elif output == "dict":
                if self.class_labels is None:
                    raise ValueError(
                        "class_labels not available — cannot produce dict output. "
                        "Load model with AERD_Weights enum to include labels."
                    )
                return {
                    self.get_class_label(i): probs[..., i]
                    for i in range(self.num_classes)
                }
            else:
                raise ValueError(f"Unknown output format: {output!r}. Use 'tensor' or 'dict'.")

    def predict_labels(
        self,
        waveform: Tensor,
        threshold: float = 0.5,
    ) -> list[list[str]] | list[list[list[str]]]:
        """
        Return class labels with probability above threshold.

        Args:
            waveform: Raw audio tensor (batch, samples) or (samples,).
            threshold: Probability threshold (default 0.5).

        Returns:
            If predictor produces per-frame output (seq, seq-spec, seq-full):
                list[list[list[str]]] — labels per frame per batch item.
            If predictor produces single output (label):
                list[list[str]] — one label list per batch item.
        """
        if self.class_labels is None:
            raise ValueError(
                "class_labels not available — cannot produce label output."
            )
        probs = self.predict(waveform, output="tensor")

        above = probs >= threshold

        has_time = self.predictor_type in ("seq", "seq-spec", "seq-full")

        if has_time:
            # probs shape: (batch, time, classes)
            return [
                [
                    [self.get_class_label(i) for i in frame.nonzero(as_tuple=True)[0].tolist()]
                    for frame in batch_item
                ]
                for batch_item in above
            ]
        else:
            # probs shape: (batch, classes)
            return [
                [self.get_class_label(i) for i in row.nonzero(as_tuple=True)[0].tolist()]
                for row in above
            ]

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        map_location: str | torch.device = "cpu",
    ) -> "AERDClassifier":
        """
        Load a pretrained AERD model from checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
            map_location: Device to load the model to.

        Returns:
            Loaded AERDClassifier model.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)

        # Extract hyperparameters
        hparams = checkpoint.get("hyper_parameters", {})
        num_classes = hparams.get("num_classes", 5)
        fbank_mean = hparams.get("fbank_mean", 14.7274)
        fbank_std = hparams.get("fbank_std", 2.9366)
        predictor_type = hparams.get("predictor", "seq")

        # Extract BEATs config if available
        beats_config = checkpoint.get("beats_config", None)

        # Extract class labels and dataset info
        class_labels = checkpoint.get("class_labels", None)
        dataset_info = checkpoint.get("dataset_info", None)

        # Create model
        model = cls(
            num_classes=num_classes,
            fbank_mean=fbank_mean,
            fbank_std=fbank_std,
            predictor_type=predictor_type,
            beats_config=beats_config,
            class_labels=class_labels,
            dataset_info=dataset_info,
        )

        # Load state dict
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)

        return model
