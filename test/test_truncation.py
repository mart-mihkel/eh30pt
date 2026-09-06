"""Tests for the shared truncation-length budgeting helper."""

from unittest.mock import MagicMock

from eh30pt.constants import UNSET_MAX_LENGTH
from eh30pt.datasets.truncation import get_max_length


def test_get_max_length_reserves_virtual_tokens_and_floors_to_pad_multiple() -> None:
    tokenizer = MagicMock(model_max_length=512)
    expected_max_length = 504

    assert get_max_length(tokenizer, n_virtual=6) == expected_max_length


def test_get_max_length_returns_none_for_unset_sentinel() -> None:
    tokenizer = MagicMock(model_max_length=UNSET_MAX_LENGTH)

    assert get_max_length(tokenizer, n_virtual=6) is None


def test_get_max_length_returns_none_above_unset_sentinel() -> None:
    tokenizer = MagicMock(model_max_length=UNSET_MAX_LENGTH * 2)

    assert get_max_length(tokenizer, n_virtual=0) is None
