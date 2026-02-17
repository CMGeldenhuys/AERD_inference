# Learning to Rumble: Automated Elephant Call Detection & Classification

[![Paper (Bioacoustics)](https://img.shields.io/badge/Paper-Bioacoustics%202025-green)](https://www.tandfonline.com/doi/full/10.1080/09524622.2025.2487099)
[![Preprint](https://img.shields.io/badge/Preprint-arXiv%202410.12082-orange)](https://arxiv.org/abs/2410.12082)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

<!-- Banner placeholder -->
<!--![Learning to Rumble Banner](assets/banner.png)-->

A deep learning system for automatically detecting, classifying, and endpointing elephant calls in continuous audio recordings. Elephants communicate using a rich vocal repertoire — from low-frequency rumbles that travel kilometres to trumpets and roars — and understanding these calls is key to monitoring their behaviour and supporting conservation efforts. This system uses an audio spectrogram transformer (AST) with transfer learning to perform frame-level call detection and classify calls into types (e.g. rumble, trumpet, roar) and sub-call categories, achieving state-of-the-art results on both Asian and African elephant vocalisation datasets.

## Getting Started

### Basic Usage (PyTorch Hub, Recommended)

PyTorch Hub automatically handles downloading and installing the required code. No additional installations beyond PyTorch are needed.

```python
import torch

# Load a pretrained model (single fold, downloads weights automatically)
model, preprocess = torch.hub.load("CMGeldenhuys/AERD_inference", "aerd")

# 10 seconds of audio at 16 kHz
audio = torch.randn(1, 16000 * 10)

# Raw probabilities — shape (1, 62, 5) for 62 time frames × 5 classes
probs = model.predict(audio, output="tensor")

# Per-frame class labels above a confidence threshold
labels = model.predict_labels(audio, threshold=0.5)
# [                          ← batch
#   [                        ← frames
#     ["rumble"],            ← frame 0: one class detected
#     ["rumble"],            ← frame 1
#     [],                    ← frame 2: nothing above threshold
#     ...
#     ["roar", "trumpet"],   ← frame 61: two classes detected
#   ]
# ]

# Probabilities as a label-keyed dictionary
label_dict = model.predict(audio, output="dict")
# {"not-call": tensor(...), "rumble": tensor(...), "roar": tensor(...),
#  "cry": tensor(...), "trumpet": tensor(...)}
```

### Ensemble Usage (pip)

Install the package to access ensemble (k-fold) prediction and all available weight variants:

```sh
pip install aerd
```

```python
import torch
from aerd import aerd_ensemble, AERD_Weights

# Load all k-fold models for ensemble prediction
models, preprocess = aerd_ensemble(weights=AERD_Weights.EV_CALL)

audio = torch.randn(1, 16000 * 10)

# Average predictions across folds
ensemble_probs = torch.stack([m.predict(audio) for m in models]).mean(dim=0)

# Available weight variants:
# AERD_Weights.EV_CALL      — 5-class elephant call type (EV dataset, 5 folds)
# AERD_Weights.EV_SUBCALL   — 8-class sub-call type (EV dataset, 5 folds)
# AERD_Weights.EV_BINARY    — binary call detection (EV dataset, 5 folds)
# AERD_Weights.LDC_POOLE    — 6-class Poole taxonomy (LDC/Asian dataset, 10 folds)
# AERD_Weights.LDC_BINARY   — binary call detection (LDC/Asian dataset, 10 folds)
```

### Requirements

- Python >= 3.11.0
- PyTorch >= 2.0.0

## Installation

### Option 1: Install via pip (Recommended)

```sh
pip install aerd
```

### Option 2: Install from source

```sh
git clone https://github.com/CMGeldenhuys/AERD_inference.git
cd AERD_inference
pip install .
```

## Datasets

The models were trained and evaluated on two annotated elephant vocalisation datasets:

- **Elephant Voices (EV)**: African elephant vocalisations with call-type and sub-call annotations.
- **Linguistic Data Consortium (LDC)**: Asian elephant vocalisations.

## Model Weights

Pre-trained model weights are available in the [GitHub Releases](https://github.com/CMGeldenhuys/AERD_inference/releases) section. Weights can be loaded automatically via PyTorch Hub or downloaded manually.

## Citation

If you use this work in your research, please cite:

```
Geldenhuys, C. M., & Niesler, T. R. (2025). Learning to rumble: Automated elephant call and sub-call
classification, detection and endpointing using deep architectures. Bioacoustics.
https://doi.org/10.1080/09524622.2025.2487099
```

```bibtex
@article{Geldenhuys2025LearningToRumble,
    author = {Geldenhuys, Christiaan M. and Niesler, Thomas R.},
    title = {Learning to rumble: automated elephant call and sub-call classification, detection and endpointing using deep architectures},
    journal = {Bioacoustics},
    volume = {34},
    number = {3},
    year = {2025},
    doi = {10.1080/09524622.2025.2487099},
    url = {https://www.tandfonline.com/doi/full/10.1080/09524622.2025.2487099}
}
```

## Contributing

We welcome contributions to improve AERD! Please feel free to submit issues, fork the repository, and create pull requests.

## Authors

- **Christiaan M. Geldenhuys** [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--0691--0235-green.svg)](https://orcid.org/0000-0003-0691-0235)

- **Thomas R. Niesler** [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--7341--1017-green.svg)](https://orcid.org/0000-0002-7341-1017)

## Acknowledgements

The authors gratefully acknowledge Telkom (South Africa) for their financial support, and the [Stellenbosch Rhasatsha high performance computing (HPC)](https://www0.sun.ac.za/hpc) facility for the compute time provided to the research presented in this work.

<!--<p align="center">
    <img src="assets/SU_logo.png" alt="Stellenbosch University Logo" height="100"/>
</p>-->

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) — a copyleft license that requires anyone who distributes the code or a derivative work to make the source available under the same terms. All code and model weights are provided as is.

---

*Published in [Bioacoustics](https://www.tandfonline.com/doi/full/10.1080/09524622.2025.2487099), Volume 34, Issue 3, 2025.*
