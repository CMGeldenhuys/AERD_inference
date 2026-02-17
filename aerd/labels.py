"""
Known label mappings for AERD models.

Label schemes follow the Poole 2011 taxonomy. The ann_ids are assigned
sequentially based on which labels appear in each dataset.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# Known label maps keyed by (dataset, variant)
LABEL_MAPS: Dict[Tuple[str, str], Dict[int, str]] = {
    # EV (vocalisation_type='call') — 5 classes, 5 folds
    ("elephant-voices", "call"): {
        0: "not-call",
        1: "rumble",
        2: "roar",
        3: "cry",
        4: "trumpet",
    },
    # EV (vocalisation_type='subcall') — 8 classes, 5 folds
    ("elephant-voices", "subcall"): {
        0: "not-call",
        1: "comment-rumble",
        2: "conflict-roar",
        3: "estrous-rumble",
        4: "female-chorus",
        5: "grumbling-rumble",
        6: "pulsated-play-trumpet",
        7: "trumpet-blast",
    },
    # LDC (annotation_scheme='poole') — 6 classes, 10 folds
    ("asian-voc", "poole"): {
        0: "not-call",
        1: "rumble",
        2: "roar",
        3: "bark",
        4: "trumpet",
        5: "squelch",
    },
    # Binary variants (any dataset)
    ("elephant-voices", "binary"): {0: "not-call", 1: "call"},
    ("asian-voc", "binary"): {0: "not-call", 1: "call"},
}

BINARY_LABELS: Dict[int, str] = {0: "not-call", 1: "call"}


def fix_target_mapping(
    target_mapping: Dict[str, int],
    is_binary: bool = False,
) -> Dict[int, str]:
    """Invert and fix a target_mapping from the training code.

    The training code's reverse mapping produces ``"call"`` at ID 1 (dict
    ordering overwrites ``"rumble"``).  For non-binary models the correct
    semantic label is ``"rumble"``, so this function corrects that.

    Args:
        target_mapping: Forward mapping ``{label_name: ann_id}`` from the
            training datamodule's ``target_mapping`` attribute.
        is_binary: When ``True`` the mapping is for a binary model and
            ``"call"`` is the correct label — no renaming is applied.

    Returns:
        Inverted mapping ``{ann_id: label_name}`` with the fix applied.
    """
    # Invert: {name: id} -> {id: name}
    inverted: Dict[int, str] = {v: k for k, v in target_mapping.items()}

    # Fix the "call" -> "rumble" collision for non-binary models
    if not is_binary and inverted.get(1) == "call":
        inverted[1] = "rumble"

    return inverted


def get_class_labels(
    dataset: str,
    variant: str,
) -> Optional[Dict[int, str]]:
    """Look up hardcoded class labels for a known (dataset, variant) pair.

    Args:
        dataset: Dataset name (``"elephant-voices"`` or ``"asian-voc"``).
        variant: Variant name (``"call"``, ``"subcall"``, ``"poole"``,
            or ``"binary"``).

    Returns:
        Label mapping ``{ann_id: label_name}`` or ``None`` if the
        combination is unknown.
    """
    return LABEL_MAPS.get((dataset, variant))
