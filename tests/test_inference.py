"""Unit tests for src/inference.py — encoder model inference functions.

Tests predict_single output format using mocks.
No GPU or network required.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestPredictSingle:
    """Verify predict_single returns correct output format."""

    def test_returns_expected_keys(self):
        from src.inference import predict_single

        # Mock model
        mock_model = MagicMock()
        logits = torch.tensor([[0.1, 0.9]])
        mock_model.return_value = MagicMock(logits=logits)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 16, dtype=torch.long),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }

        result = predict_single(
            "test text", mock_model, mock_tokenizer, device="cpu"
        )
        assert "label" in result
        assert "label_idx" in result
        assert "probability" in result
        assert "all_probs" in result

    def test_label_idx_is_argmax(self):
        from src.inference import predict_single

        mock_model = MagicMock()
        # Class 1 has highest logit
        logits = torch.tensor([[0.1, 0.9]])
        mock_model.return_value = MagicMock(logits=logits)

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 16, dtype=torch.long),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }

        result = predict_single(
            "test", mock_model, mock_tokenizer, device="cpu",
            label_map={0: "none", 1: "hate"},
        )
        assert result["label_idx"] == 1
        assert result["label"] == "hate"

    def test_probability_between_0_and_1(self):
        from src.inference import predict_single

        mock_model = MagicMock()
        logits = torch.tensor([[2.0, -1.0, 0.5]])
        mock_model.return_value = MagicMock(logits=logits)

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 16, dtype=torch.long),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }

        result = predict_single(
            "text", mock_model, mock_tokenizer, device="cpu",
            label_map={0: "a", 1: "b", 2: "c"},
        )
        assert 0.0 <= result["probability"] <= 1.0

    def test_default_label_map(self):
        from src.inference import predict_single

        mock_model = MagicMock()
        logits = torch.tensor([[0.9, 0.1]])
        mock_model.return_value = MagicMock(logits=logits)

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 16, dtype=torch.long),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }

        result = predict_single("text", mock_model, mock_tokenizer, device="cpu")
        # Default label_map: {0: "none", 1: "hate"}
        assert result["label"] in ("none", "hate")


class TestLoadModelAndTokenizer:
    """Verify load_model_and_tokenizer argument validation."""

    def test_raises_without_model_path_or_name(self):
        from src.inference import load_model_and_tokenizer

        with pytest.raises(ValueError, match="Either"):
            load_model_and_tokenizer(model_path=None, model_name=None)
