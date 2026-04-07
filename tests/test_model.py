"""Unit tests for src/model.py — model building utilities.

Tests build_model and load_trained_model function signatures and error handling.
Uses mocks to avoid downloading real models.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestBuildModel:
    """Verify build_model function contract."""

    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForSequenceClassification")
    def test_returns_model_and_tokenizer(self, mock_model_cls, mock_tok_cls):
        from src.model import build_model

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        model, tokenizer = build_model("test/model", num_labels=3, device="cpu")
        assert model is not None
        assert tokenizer is not None

    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForSequenceClassification")
    def test_model_moved_to_device(self, mock_model_cls, mock_tok_cls):
        from src.model import build_model

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model

        build_model("test/model", num_labels=2, device="cpu")
        mock_model.to.assert_called_once_with("cpu")

    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForSequenceClassification")
    def test_num_labels_passed(self, mock_model_cls, mock_tok_cls):
        from src.model import build_model

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model

        build_model("test/model", num_labels=5, device="cpu")
        mock_model_cls.from_pretrained.assert_called_once_with("test/model", num_labels=5)


class TestLoadTrainedModel:
    """Verify load_trained_model function contract."""

    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForSequenceClassification")
    def test_returns_model_and_tokenizer(self, mock_model_cls, mock_tok_cls):
        from src.model import load_trained_model

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model

        model, tokenizer = load_trained_model("path/to/model", device="cpu")
        assert model is not None
        assert tokenizer is not None

    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForSequenceClassification")
    def test_model_set_to_eval(self, mock_model_cls, mock_tok_cls):
        from src.model import load_trained_model

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model

        load_trained_model("path/to/model", device="cpu")
        mock_model.eval.assert_called_once()
