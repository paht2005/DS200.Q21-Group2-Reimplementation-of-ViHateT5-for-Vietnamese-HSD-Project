"""Unit tests for src/t5_data_collator.py — T5 Span Corruption utilities.

Tests compute_t5_input_and_target_lengths and DataCollatorForT5MLM logic.
No GPU or network required.
"""

import numpy as np
import pytest


class TestComputeT5InputTargetLengths:
    """Verify input/target length computation for span corruption."""

    def test_returns_two_integers(self):
        from src.t5_data_collator import compute_t5_input_and_target_lengths

        tokens_len, targets_len = compute_t5_input_and_target_lengths(
            inputs_length=512, noise_density=0.15, mean_noise_span_length=3.0
        )
        assert isinstance(tokens_len, int)
        assert isinstance(targets_len, int)

    def test_tokens_length_greater_than_input(self):
        """Raw tokens should be >= input length to fill after masking."""
        from src.t5_data_collator import compute_t5_input_and_target_lengths

        tokens_len, _ = compute_t5_input_and_target_lengths(
            inputs_length=256, noise_density=0.15, mean_noise_span_length=3.0
        )
        assert tokens_len >= 256

    def test_targets_length_positive(self):
        from src.t5_data_collator import compute_t5_input_and_target_lengths

        _, targets_len = compute_t5_input_and_target_lengths(
            inputs_length=256, noise_density=0.15, mean_noise_span_length=3.0
        )
        assert targets_len > 0

    def test_higher_noise_density_increases_targets(self):
        from src.t5_data_collator import compute_t5_input_and_target_lengths

        _, targets_low = compute_t5_input_and_target_lengths(
            inputs_length=256, noise_density=0.10, mean_noise_span_length=3.0
        )
        _, targets_high = compute_t5_input_and_target_lengths(
            inputs_length=256, noise_density=0.30, mean_noise_span_length=3.0
        )
        assert targets_high > targets_low

    def test_noise_density_half(self):
        """Special case: noise_density=0.5 should handle the minor hack."""
        from src.t5_data_collator import compute_t5_input_and_target_lengths

        tokens_len, targets_len = compute_t5_input_and_target_lengths(
            inputs_length=256, noise_density=0.5, mean_noise_span_length=3.0
        )
        assert tokens_len > 0
        assert targets_len > 0


class TestShiftTokensRight:
    """Verify decoder input preparation."""

    def test_shift_tokens_right(self):
        from src.t5_data_collator import shift_tokens_right

        input_ids = np.array([[10, 20, 30, 40]])
        shifted = shift_tokens_right(input_ids, pad_token_id=0, decoder_start_token_id=99)
        assert shifted[0, 0] == 99
        assert shifted[0, 1] == 10
        assert shifted[0, 2] == 20
        assert shifted[0, 3] == 30

    def test_shift_replaces_negative_100_with_pad(self):
        from src.t5_data_collator import shift_tokens_right

        input_ids = np.array([[10, -100, 30, -100]])
        shifted = shift_tokens_right(input_ids, pad_token_id=0, decoder_start_token_id=99)
        # After shifting: [99, 10, -100, 30] -> -100 replaced with pad (0)
        assert shifted[0, 2] == 0

    def test_batch_dimension(self):
        from src.t5_data_collator import shift_tokens_right

        input_ids = np.array([[1, 2, 3], [4, 5, 6]])
        shifted = shift_tokens_right(input_ids, pad_token_id=0, decoder_start_token_id=0)
        assert shifted.shape == (2, 3)
