"""Tests for AERDClassifier interfaces."""

import pytest
import torch

from aerd.model import AERDClassifier


class TestForward:
    def test_forward_seq_shape(self, ev_call_model, dummy_audio):
        logits = ev_call_model(dummy_audio)
        B = dummy_audio.shape[0]
        C = ev_call_model.num_classes
        assert logits.ndim == 3
        assert logits.shape[0] == B
        assert logits.shape[2] == C
        assert logits.shape[1] > 0  # T > 0

    def test_forward_label_shape(self, label_model, dummy_audio):
        logits = label_model(dummy_audio)
        B = dummy_audio.shape[0]
        C = label_model.num_classes
        assert logits.shape == (B, C)

    def test_forward_single_waveform(self, ev_call_model, dummy_audio_1d):
        logits = ev_call_model(dummy_audio_1d)
        assert logits.shape[0] == 1  # batch dim added

    def test_forward_return_features(self, ev_call_model, dummy_audio):
        result = ev_call_model(dummy_audio, return_features=True)
        assert isinstance(result, tuple) and len(result) == 2
        logits, features = result
        assert isinstance(logits, torch.Tensor)
        assert isinstance(features, torch.Tensor)


class TestPreprocess:
    def test_preprocess_output(self, ev_call_model, dummy_audio):
        fbank, mask = ev_call_model.preprocess(dummy_audio)
        B = dummy_audio.shape[0]
        assert fbank.ndim == 3
        assert fbank.shape[0] == B
        assert fbank.shape[2] == 128
        assert mask.ndim == 2
        assert mask.dtype == torch.bool


class TestPredict:
    def test_predict_tensor_output(self, ev_call_model, dummy_audio):
        probs = ev_call_model.predict(dummy_audio, output="tensor")
        assert isinstance(probs, torch.Tensor)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_dict_output(self, ev_call_model, dummy_audio):
        result = ev_call_model.predict(dummy_audio, output="dict")
        assert isinstance(result, dict)
        assert set(result.keys()) == set(ev_call_model.class_labels.values())
        for v in result.values():
            assert isinstance(v, torch.Tensor)

    def test_predict_dict_no_labels_raises(self, model_no_labels, dummy_audio):
        with pytest.raises(ValueError, match="class_labels"):
            model_no_labels.predict(dummy_audio, output="dict")

    def test_predict_invalid_output_raises(self, ev_call_model, dummy_audio):
        with pytest.raises(ValueError, match="foo"):
            ev_call_model.predict(dummy_audio, output="foo")


class TestPredictLabels:
    def test_predict_labels_seq_type(self, ev_call_model, dummy_audio):
        result = ev_call_model.predict_labels(dummy_audio)
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], list)
        # inner elements are strings (if any)
        for frame in result[0]:
            for label in frame:
                assert isinstance(label, str)

    def test_predict_labels_label_type(self, label_model, dummy_audio):
        result = label_model.predict_labels(dummy_audio)
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        # elements are strings directly (no frame nesting)
        for label in result[0]:
            assert isinstance(label, str)

    def test_predict_labels_no_labels_raises(self, model_no_labels, dummy_audio):
        with pytest.raises(ValueError, match="class_labels"):
            model_no_labels.predict_labels(dummy_audio)

    def test_predict_labels_threshold_zero(self, ev_call_model, dummy_audio):
        result = ev_call_model.predict_labels(dummy_audio, threshold=0.0)
        all_labels = set(ev_call_model.class_labels.values())
        # Every frame should contain all labels
        for batch_item in result:
            for frame in batch_item:
                assert set(frame) == all_labels

    def test_predict_labels_threshold_one(self, ev_call_model, dummy_audio):
        result = ev_call_model.predict_labels(dummy_audio, threshold=1.01)
        # sigmoid output < 1.01 always, so all lists empty
        for batch_item in result:
            for frame in batch_item:
                assert frame == []

    def test_predict_labels_filter_excludes_label(self, ev_call_model, dummy_audio):
        result = ev_call_model.predict_labels(dummy_audio, threshold=0.0, filter={"not-call"})
        for batch_item in result:
            for frame in batch_item:
                assert "not-call" not in frame

    def test_predict_labels_filter_none_default(self, ev_call_model, dummy_audio):
        result = ev_call_model.predict_labels(dummy_audio, threshold=0.0)
        # With threshold=0 all labels appear, including not-call
        found = any("not-call" in frame for batch_item in result for frame in batch_item)
        assert found

    def test_predict_labels_filter_all_returns_empty(self, ev_call_model, dummy_audio):
        all_labels = set(ev_call_model.class_labels.values())
        result = ev_call_model.predict_labels(dummy_audio, threshold=0.0, filter=all_labels)
        for batch_item in result:
            for frame in batch_item:
                assert frame == []


class TestClassLabelNormalization:
    """Test that class_labels accepts both forward and reverse mappings."""

    REVERSE = {0: "not-call", 1: "rumble", 2: "roar"}
    FORWARD = {"not-call": 0, "rumble": 1, "roar": 2}

    def test_reverse_mapping_stored_as_is(self):
        model = AERDClassifier(num_classes=3, class_labels=self.REVERSE)
        assert model.class_labels == self.REVERSE
        assert model.class_indexes == self.FORWARD

    def test_forward_mapping_inverted(self):
        model = AERDClassifier(num_classes=3, class_labels=self.FORWARD)
        assert model.class_labels == self.REVERSE
        assert model.class_indexes == self.FORWARD

    def test_none_labels(self):
        model = AERDClassifier(num_classes=3, class_labels=None)
        assert model.class_labels is None
        assert model.class_indexes is None

    def test_get_class_label(self):
        model = AERDClassifier(num_classes=3, class_labels=self.REVERSE)
        assert model.get_class_label(0) == "not-call"
        assert model.get_class_label(1) == "rumble"
        assert model.get_class_label(2) == "roar"

    def test_get_class_index(self):
        model = AERDClassifier(num_classes=3, class_labels=self.REVERSE)
        assert model.get_class_index("not-call") == 0
        assert model.get_class_index("rumble") == 1
        assert model.get_class_index("roar") == 2

    def test_get_class_label_no_labels_raises(self):
        model = AERDClassifier(num_classes=3, class_labels=None)
        with pytest.raises(ValueError, match="class_labels"):
            model.get_class_label(0)

    def test_get_class_index_no_labels_raises(self):
        model = AERDClassifier(num_classes=3, class_labels=None)
        with pytest.raises(ValueError, match="class_indexes"):
            model.get_class_index("rumble")

    def test_predict_dict_with_forward_mapping(self, dummy_audio):
        model = AERDClassifier(num_classes=3, class_labels=self.FORWARD)
        result = model.predict(dummy_audio, output="dict")
        assert set(result.keys()) == {"not-call", "rumble", "roar"}

    def test_predict_labels_with_forward_mapping(self, dummy_audio):
        model = AERDClassifier(num_classes=3, class_labels=self.FORWARD)
        result = model.predict_labels(dummy_audio, threshold=0.0)
        expected = {"not-call", "rumble", "roar"}
        for batch_item in result:
            for frame in batch_item:
                assert set(frame) == expected


class TestMisc:
    def test_sample_rate(self):
        assert AERDClassifier.SAMPLE_RATE == 16_000
