"""Smoke test for installed aerd package.

Runs outside the repo (--no-project) to verify the built distribution
includes all necessary files and the public API is importable.
"""

import sys


def main() -> None:
    # Core imports
    from aerd import (
        AERDClassifier,
        AERD_Weights,
        aerd,
        aerd_ensemble,
        LABEL_MAPS,
        get_class_labels,
    )

    # Model instantiation (random weights, no download)
    model = AERDClassifier(num_classes=5)
    assert model.SAMPLE_RATE == 16_000

    # Weights enum is accessible
    assert AERD_Weights.EV_CALL.num_classes == 5

    # Hub entry point works without weights
    m, preprocess = aerd(weights=None)
    assert callable(preprocess)

    # Labels are present
    assert ("elephant-voices", "call") in LABEL_MAPS
    assert get_class_labels("elephant-voices", "call") is not None

    # Forward pass with random input
    import torch

    audio = torch.randn(1, 16000)
    out = m(audio)
    assert isinstance(out, torch.Tensor)

    print("smoke test passed", file=sys.stderr)


if __name__ == "__main__":
    main()
