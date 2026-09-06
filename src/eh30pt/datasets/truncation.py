"""Shared truncation-length budgeting for prompt-tuning virtual tokens."""

from typing import TYPE_CHECKING

from eh30pt.constants import PAD_MULTIPLE, UNSET_MAX_LENGTH

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


def get_max_length(
    tokenizer: PreTrainedTokenizerFast,
    n_virtual: int,
) -> int | None:
    """
    Return the truncation length that leaves room for virtual tokens.

    Returns None if the tokenizer has no real length limit configured, since
    passing its sentinel value as an explicit `max_length` would overflow the
    tokenizers' Rust backend.
    """
    if tokenizer.model_max_length >= UNSET_MAX_LENGTH:
        return None

    budget = tokenizer.model_max_length - n_virtual
    return (budget // PAD_MULTIPLE) * PAD_MULTIPLE
