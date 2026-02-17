"""Tests for AERDClassifier interfaces."""

import pytest
import torch

from aerd_inference.model import AERDClassifier


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


class TestMisc:
    def test_sample_rate(self):
        assert AERDClassifier.SAMPLE_RATE == 16_000
