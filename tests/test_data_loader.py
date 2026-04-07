"""Unit tests for src/data_loader.py — TextDataset and dataset routing.

Tests TextDataset construction, tokenization, and load_dataset_by_name routing.
Uses mock tokenizers to avoid downloading real models.
"""

import numpy as np
import pytest
import torch


class _FakeTokenizer:
    """Minimal tokenizer stub that returns fixed-size tensors."""

    def __init__(self, vocab_size=100, name_or_path="fake/tokenizer"):
        self.vocab_size = vocab_size
        self.name_or_path = name_or_path

    def __call__(self, text, max_length=256, padding="max_length",
                 truncation=True, return_tensors="pt"):
        ids = torch.randint(1, self.vocab_size, (1, max_length))
        mask = torch.ones(1, max_length, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}


class TestTextDataset:
    """Verify TextDataset construction and __getitem__."""

    def test_length(self):
        from src.data_loader import TextDataset

        texts = ["hello", "world", "test"]
        labels = [0, 1, 0]
        ds = TextDataset(texts, labels, _FakeTokenizer(), max_length=32)
        assert len(ds) == 3

    def test_getitem_returns_expected_keys(self):
        from src.data_loader import TextDataset

        ds = TextDataset(["sample"], [1], _FakeTokenizer(), max_length=16)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_getitem_tensor_shapes(self):
        from src.data_loader import TextDataset

        max_len = 64
        ds = TextDataset(["sample"], [0], _FakeTokenizer(), max_length=max_len)
        item = ds[0]
        assert item["input_ids"].shape == (max_len,)
        assert item["attention_mask"].shape == (max_len,)
        assert item["labels"].shape == ()

    def test_label_is_long_tensor(self):
        from src.data_loader import TextDataset

        ds = TextDataset(["sample"], [2], _FakeTokenizer(), max_length=16)
        item = ds[0]
        assert item["labels"].dtype == torch.long

    def test_label_value_preserved(self):
        from src.data_loader import TextDataset

        ds = TextDataset(["a", "b"], [0, 2], _FakeTokenizer(), max_length=16)
        assert ds[0]["labels"].item() == 0
        assert ds[1]["labels"].item() == 2

    def test_empty_dataset(self):
        from src.data_loader import TextDataset

        ds = TextDataset([], [], _FakeTokenizer(), max_length=16)
        assert len(ds) == 0

    def test_word_seg_requires_underthesea(self):
        """When use_word_seg=True but underthesea is not installed, should raise."""
        from src.data_loader import TextDataset
        import src.data_loader as dl_mod

        if dl_mod.word_tokenize is not None:
            pytest.skip("underthesea is installed, cannot test ImportError path")

        with pytest.raises(ImportError, match="underthesea"):
            TextDataset(["test"], [0], _FakeTokenizer(), max_length=16, use_word_seg=True)


class TestLoadDatasetByName:
    """Verify load_dataset_by_name routing logic (no actual downloads)."""

    def test_unknown_dataset_raises(self):
        from src.data_loader import load_dataset_by_name

        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset_by_name("nonexistent_dataset_xyz")

    def test_known_datasets_are_registered(self):
        """Verify the predefined dataset names are present in the router."""
        from src.data_loader import load_dataset_by_name
        import inspect

        source = inspect.getsource(load_dataset_by_name)
        for name in ["ViHSD", "ViCTSD", "ViHOS", "ViHSD_processed"]:
            assert name in source, f"Dataset '{name}' not found in load_dataset_by_name"

    def test_voz_hsd_route_recognized(self):
        """VOZ-HSD_2M should be recognized (even if download fails)."""
        from src.data_loader import load_dataset_by_name
        import inspect

        source = inspect.getsource(load_dataset_by_name)
        assert "VOZ-HSD_2M" in source
