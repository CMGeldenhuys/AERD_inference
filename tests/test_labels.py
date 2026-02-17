"""Tests for LABEL_MAPS and get_class_labels."""

import pytest

from aerd_inference.labels import LABEL_MAPS, get_class_labels


_EXPECTED_KEYS = [
    ("elephant-voices", "call"),
    ("elephant-voices", "subcall"),
    ("elephant-voices", "binary"),
    ("asian-voc", "poole"),
    ("asian-voc", "binary"),
]


def test_label_maps_expected_keys():
    for key in _EXPECTED_KEYS:
        assert key in LABEL_MAPS, f"Missing key {key}"


@pytest.mark.parametrize("key", _EXPECTED_KEYS, ids=lambda k: f"{k[0]}_{k[1]}")
def test_label_maps_contiguous_indices(key):
    mapping = LABEL_MAPS[key]
    expected_indices = list(range(len(mapping)))
    assert sorted(mapping.keys()) == expected_indices


def test_get_class_labels_known():
    result = get_class_labels("elephant-voices", "call")
    assert result is not None
    assert isinstance(result, dict)
    assert result == LABEL_MAPS[("elephant-voices", "call")]


def test_get_class_labels_unknown():
    result = get_class_labels("unknown", "unknown")
    assert result is None
