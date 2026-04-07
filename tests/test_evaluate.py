"""Unit tests for src/evaluate.py — T5 evaluation helper functions.

Tests label mapping and span processing utilities.
No GPU or network required.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# evaluate.py uses bare imports (e.g., "from data_loader import ..."),
# so we need src/ on sys.path for it to import correctly.
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


class TestMapDataViHSD:
    """Verify ViHSD label mapping and prefix generation."""

    def test_label_mapping(self):
        from evaluate import map_data_vihsd

        df = pd.DataFrame({"free_text": ["hello", "world"], "label_id": [0, 2]})
        result = map_data_vihsd(df)
        assert result["target"].iloc[0] == "CLEAN"
        assert result["target"].iloc[1] == "HATE"

    def test_source_prefix(self):
        from evaluate import map_data_vihsd

        df = pd.DataFrame({"free_text": ["test text"], "label_id": [1]})
        result = map_data_vihsd(df)
        assert result["source"].iloc[0].startswith("hate-speech-detection: ")

    def test_output_columns(self):
        from evaluate import map_data_vihsd

        df = pd.DataFrame({"free_text": ["t"], "label_id": [0], "extra": [1]})
        result = map_data_vihsd(df)
        assert list(result.columns) == ["source", "target"]

    def test_all_label_ids(self):
        from evaluate import map_data_vihsd

        df = pd.DataFrame({
            "free_text": ["a", "b", "c"],
            "label_id": [0, 1, 2],
        })
        result = map_data_vihsd(df)
        assert list(result["target"]) == ["CLEAN", "OFFENSIVE", "HATE"]


class TestMapDataViCTSD:
    """Verify ViCTSD label mapping."""

    def test_label_mapping(self):
        from evaluate import map_data_victsd

        df = pd.DataFrame({"Comment": ["good", "bad"], "Toxicity": [0, 1]})
        result = map_data_victsd(df)
        assert result["target"].iloc[0] == "NONE"
        assert result["target"].iloc[1] == "TOXIC"

    def test_source_prefix(self):
        from evaluate import map_data_victsd

        df = pd.DataFrame({"Comment": ["test"], "Toxicity": [0]})
        result = map_data_victsd(df)
        assert result["source"].iloc[0].startswith("toxic-speech-detection: ")


class TestProcessSpans:
    """Verify span index processing."""

    def test_empty_string(self):
        from evaluate import process_spans

        assert process_spans("[]") == []

    def test_empty_value(self):
        from evaluate import process_spans

        assert process_spans("") == []

    def test_none_value(self):
        from evaluate import process_spans

        assert process_spans(None) == []

    def test_single_span(self):
        from evaluate import process_spans

        result = process_spans("[1, 2, 3]")
        assert result == [[1, 2, 3]]

    def test_multiple_spans(self):
        from evaluate import process_spans

        result = process_spans("[1, 2, 3, 10, 11]")
        assert len(result) == 2
        assert result[0] == [1, 2, 3]
        assert result[1] == [10, 11]

    def test_single_element(self):
        from evaluate import process_spans

        result = process_spans("[5]")
        assert result == [[5]]


class TestAddTags:
    """Verify hate span tag insertion."""

    def test_no_spans(self):
        from evaluate import add_tags

        assert add_tags("hello world", "[]") == "hello world"

    def test_empty_spans(self):
        from evaluate import add_tags

        assert add_tags("hello world", "") == "hello world"

    def test_none_spans(self):
        from evaluate import add_tags

        assert add_tags("hello world", None) == "hello world"
