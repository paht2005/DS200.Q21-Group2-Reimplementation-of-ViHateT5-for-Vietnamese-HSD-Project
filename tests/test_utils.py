"""Unit tests for src/utils.py — training and evaluation utility functions.

Tests compute_metrics, set_seed reproducibility.
No GPU or network required.
"""

import numpy as np
import pytest
import torch


class TestComputeMetrics:
    """Verify compute_metrics returns correct metric values."""

    def test_perfect_predictions(self):
        from src.utils import compute_metrics

        preds = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 2, 0, 1])
        metrics = compute_metrics(preds, labels)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["weighted_f1"] == 1.0

    def test_all_wrong_predictions(self):
        from src.utils import compute_metrics

        preds = np.array([1, 0, 1, 0])
        labels = np.array([0, 1, 0, 1])
        metrics = compute_metrics(preds, labels)
        assert metrics["accuracy"] == 0.0

    def test_partial_predictions(self):
        from src.utils import compute_metrics

        preds = np.array([0, 1, 0, 1])
        labels = np.array([0, 1, 1, 0])
        metrics = compute_metrics(preds, labels)
        assert metrics["accuracy"] == 0.5
        assert 0.0 < metrics["macro_f1"] < 1.0
        assert 0.0 < metrics["weighted_f1"] < 1.0

    def test_returns_dict_with_expected_keys(self):
        from src.utils import compute_metrics

        preds = np.array([0, 1])
        labels = np.array([0, 1])
        metrics = compute_metrics(preds, labels)
        assert set(metrics.keys()) == {"accuracy", "macro_f1", "weighted_f1"}

    def test_binary_classification(self):
        from src.utils import compute_metrics

        preds = np.array([0, 0, 1, 1, 0])
        labels = np.array([0, 1, 1, 1, 0])
        metrics = compute_metrics(preds, labels)
        assert metrics["accuracy"] == 4 / 5

    def test_single_class_predictions(self):
        from src.utils import compute_metrics

        preds = np.array([0, 0, 0, 0])
        labels = np.array([0, 0, 0, 0])
        metrics = compute_metrics(preds, labels)
        assert metrics["accuracy"] == 1.0


class TestSetSeed:
    """Verify set_seed produces reproducible results."""

    def test_same_seed_same_output(self):
        from src.utils import set_seed

        set_seed(42)
        t1 = torch.rand(10)
        r1 = np.random.rand(10)

        set_seed(42)
        t2 = torch.rand(10)
        r2 = np.random.rand(10)

        assert torch.allclose(t1, t2)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seed_different_output(self):
        from src.utils import set_seed

        set_seed(1)
        t1 = torch.rand(10)

        set_seed(2)
        t2 = torch.rand(10)

        assert not torch.allclose(t1, t2)
