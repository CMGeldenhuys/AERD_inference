"""Tests for AERD_Weights enum, URL generation, and aliases."""

import pytest

from aerd_inference.weights import AERD_Weights


_CANONICAL_VARIANTS = [
    AERD_Weights.EV_CALL,
    AERD_Weights.EV_SUBCALL,
    AERD_Weights.EV_BINARY,
    AERD_Weights.LDC_POOLE,
    AERD_Weights.LDC_BINARY,
]


@pytest.mark.parametrize("w", _CANONICAL_VARIANTS, ids=lambda w: w.name)
def test_weight_variants_have_required_properties(w):
    assert isinstance(w.num_classes, int) and w.num_classes > 0
    assert isinstance(w.num_folds, int) and w.num_folds > 0
    assert isinstance(w.class_labels, dict) and len(w.class_labels) > 0
    assert isinstance(w.file_hashes, dict) and len(w.file_hashes) > 0
    assert isinstance(w.meta, dict)
    assert isinstance(w.version, str) and len(w.version) > 0
    assert isinstance(w.url_template, str) and "{version}" in w.url_template


@pytest.mark.parametrize("w", _CANONICAL_VARIANTS, ids=lambda w: w.name)
def test_file_hashes_length_matches_num_folds(w):
    assert len(w.file_hashes) == w.num_folds


@pytest.mark.parametrize("w", _CANONICAL_VARIANTS, ids=lambda w: w.name)
def test_get_url_contains_hash(w):
    fold = 1
    url = w.get_url(fold)
    expected_hash = w.file_hashes[fold]
    assert len(expected_hash) == 8
    assert expected_hash in url


def test_get_url_version_override():
    w = AERD_Weights.EV_CALL
    url = w.get_url(fold=1, version="2")
    assert "/v2/" in url


def test_aliases_resolve_to_canonical():
    assert AERD_Weights.DEFAULT_EV is AERD_Weights.EV_CALL
    assert AERD_Weights.DEFAULT_LDC is AERD_Weights.LDC_POOLE
    assert AERD_Weights.ENSEMBLE_EV is AERD_Weights.EV_CALL
    assert AERD_Weights.ENSEMBLE_LDC is AERD_Weights.LDC_POOLE


@pytest.mark.parametrize("w", _CANONICAL_VARIANTS, ids=lambda w: w.name)
def test_url_base_points_to_inference_repo(w):
    url = w.get_url(fold=1)
    assert "AERD_inference" in url
