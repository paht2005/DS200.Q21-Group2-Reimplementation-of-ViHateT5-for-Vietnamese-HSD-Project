"""Unit tests for src/config.py — TrainConfig dataclass.

Validates construction, defaults, serialization, and seed behaviour.
No GPU or network required.
"""

from pathlib import Path

import pytest
import torch


class TestTrainConfigDefaults:
    """Verify TrainConfig default values are set correctly."""

    def test_default_model_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.config import TrainConfig

        cfg = TrainConfig(dataset_name="ViHSD")
        assert cfg.model_name == "vinai/phobert-base"

    def test_default_hyperparameters(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.config import TrainConfig

        cfg = TrainConfig(dataset_name="ViHSD")
        assert cfg.max_length == 256
        assert cfg.batch_size == 16
        assert cfg.epochs == 10
        assert cfg.learning_rate == 2e-5
        assert cfg.weight_decay == 0.01
        assert cfg.warmup_ratio == 0.1
        assert cfg.patience == 3
        assert cfg.seed == 42

    def test_device_is_string(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.config import TrainConfig

        cfg = TrainConfig(dataset_name="ViHSD")
        assert cfg.device in ("cuda", "cpu")


class TestTrainConfigOutputDir:
    """Verify output directory auto-creation logic."""

    def test_auto_output_dir_contains_dataset_and_model(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.config import TrainConfig

        cfg = TrainConfig(dataset_name="ViCTSD", model_name="uitnlp/visobert")
        dir_name = cfg.output_dir.name
        assert "ViCTSD" in dir_name
        assert "visobert" in dir_name

    def test_auto_output_dir_is_created(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.config import TrainConfig

        cfg = TrainConfig(dataset_name="ViHSD")
        assert cfg.output_dir.is_dir()

    def test_explicit_output_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.config import TrainConfig

        custom_dir = tmp_path / "my_output"
        cfg = TrainConfig(dataset_name="ViHSD", output_dir=custom_dir)
        assert cfg.output_dir == custom_dir
        assert custom_dir.is_dir()


class TestTrainConfigSerialization:
    """Verify to_dict() round-trip."""

    def test_to_dict_returns_dict(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.config import TrainConfig

        cfg = TrainConfig(dataset_name="ViHSD")
        d = cfg.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_keys(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.config import TrainConfig

        cfg = TrainConfig(dataset_name="ViHSD")
        d = cfg.to_dict()
        expected_keys = {
            "dataset_name", "model_name", "max_length", "batch_size",
            "epochs", "learning_rate", "weight_decay", "warmup_ratio",
            "patience", "seed", "device", "output_dir",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.config import TrainConfig

        cfg = TrainConfig(dataset_name="ViHOS", learning_rate=1e-4, epochs=5)
        d = cfg.to_dict()
        assert d["dataset_name"] == "ViHOS"
        assert d["learning_rate"] == 1e-4
        assert d["epochs"] == 5

    def test_output_dir_serialized_as_string(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.config import TrainConfig

        cfg = TrainConfig(dataset_name="ViHSD")
        d = cfg.to_dict()
        assert isinstance(d["output_dir"], str)


class TestTrainConfigSeed:
    """Verify seed reproducibility."""

    def test_seed_sets_torch_manual_seed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.config import TrainConfig

        TrainConfig(dataset_name="ViHSD", seed=123)
        t1 = torch.rand(5)
        TrainConfig(dataset_name="ViHSD", seed=123)
        t2 = torch.rand(5)
        assert torch.allclose(t1, t2), "Same seed should produce same random tensors"

    def test_different_seed_gives_different_output(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.config import TrainConfig

        TrainConfig(dataset_name="ViHSD", seed=1)
        t1 = torch.rand(5)
        TrainConfig(dataset_name="ViHSD", seed=2)
        t2 = torch.rand(5)
        assert not torch.allclose(t1, t2), "Different seeds should produce different tensors"
