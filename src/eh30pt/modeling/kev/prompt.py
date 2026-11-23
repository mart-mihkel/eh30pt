"""Kev prompt encoding: one causal row per decision question."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedTokenizerFast

    from eh30pt.modeling.kev.records import KevRecord

_special_pattern = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class ContextOverflowError(ValueError):
    """A decision row does not fit the configured state or branch budget."""


@dataclass(frozen=True)
class Delimiters:
    """Token ids of the five delimiters the pointer head reads."""

    state: int
    instruction: int
    option_open: int
    option_close: int
    decide: int

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizerFast) -> Delimiters:
        """Resolve the delimiter ids, rejecting backbones without these tokens."""
        tokens = {
            "state": "<|fim_prefix|>",
            "instruction": "<|fim_middle|>",
            "option_open": "<|box_start|>",
            "option_close": "<|box_end|>",
            "decide": "<|fim_suffix|>",
        }
        vocab = tokenizer.get_vocab()
        missing = sorted(token for token in tokens.values() if token not in vocab)
        if missing:
            msg = f"tokenizer has no decision delimiters: {', '.join(missing)}"
            raise ValueError(msg)

        resolved: dict[str, int] = {}
        for name, token in tokens.items():
            token_id = tokenizer.convert_tokens_to_ids(token)
            if not isinstance(token_id, int):
                msg = f"tokenizer did not resolve '{token}' to a single id"
                raise TypeError(msg)
            resolved[name] = token_id

        return cls(
            state=resolved["state"],
            instruction=resolved["instruction"],
            option_open=resolved["option_open"],
            option_close=resolved["option_close"],
            decide=resolved["decide"],
        )


@dataclass(frozen=True)
class EncodedKevRow:
    """Token ids of one question, with the offsets the pointer head reads."""

    input_ids: tuple[int, ...]
    position_ids: tuple[int, ...]
    decide_index: int
    option_indices: tuple[int, ...]
    state_tokens: int
    truncated: bool

    @property
    def length(self) -> int:
        """Return the number of tokens in the row."""
        return len(self.input_ids)

    @property
    def option_count(self) -> int:
        """Return the number of options scored for this question."""
        return len(self.option_indices)


class EncodedSample(NamedTuple):
    """A decision row paired with its gold option index."""

    encoded: EncodedKevRow
    label: int


def user_tokens(tokenizer: PreTrainedTokenizerFast, text: str) -> list[int]:
    """Tokenize caller text without letting it forge delimiter tokens."""
    escaped = _special_pattern.sub(r"<¦\1¦>", text)
    return tokenizer(escaped, add_special_tokens=False).input_ids


def encode_kev_row(
    tokenizer: PreTrainedTokenizerFast,
    *,
    state: str,
    instructions: str,
    options: Sequence[str],
    max_state: int = 384,
    max_branch: int = 1024,
) -> EncodedKevRow:
    """Encode one question as a single causal row in the Kev prompt layout."""
    delimiters = Delimiters.from_tokenizer(tokenizer)
    if not options:
        msg = "a decision question needs at least one option"
        raise ValueError(msg)

    state_ids = [delimiters.state, *user_tokens(tokenizer, state)]
    if len(state_ids) > max_state:
        msg = f"state exceeds {max_state} tokens: {len(state_ids)}"
        raise ContextOverflowError(msg)

    branch: list[int] = [delimiters.instruction, *user_tokens(tokenizer, instructions)]
    option_indices: list[int] = []
    for option in options:
        branch.append(delimiters.option_open)
        branch.extend(user_tokens(tokenizer, option))
        branch.append(delimiters.option_close)
        option_indices.append(len(state_ids) + len(branch) - 1)
    branch.append(delimiters.decide)

    input_ids = [*state_ids, *branch]
    if len(input_ids) > max_branch:
        msg = f"question row exceeds {max_branch} tokens: {len(input_ids)}"
        raise ContextOverflowError(msg)

    return EncodedKevRow(
        input_ids=tuple(input_ids),
        position_ids=tuple(range(len(input_ids))),
        decide_index=len(input_ids) - 1,
        option_indices=tuple(option_indices),
        state_tokens=len(state_ids),
        truncated=False,
    )


def encode_records(
    tokenizer: PreTrainedTokenizerFast,
    records: Sequence[KevRecord],
    *,
    max_state: int = 384,
    max_branch: int = 1024,
) -> list[EncodedSample]:
    """Encode labelled records, dropping the ones over the configured budget."""
    samples: list[EncodedSample] = []
    overflowed = 0
    for record in records:
        try:
            encoded = encode_kev_row(
                tokenizer,
                state=record.state,
                instructions=record.instructions,
                options=record.options,
                max_state=max_state,
                max_branch=max_branch,
            )
        except ContextOverflowError:
            overflowed += 1
            continue

        samples.append(EncodedSample(encoded=encoded, label=record.label))

    if overflowed:
        logger.warning(
            "dropped %d of %d decision records over the %d/%d token budget",
            overflowed,
            len(records),
            max_state,
            max_branch,
        )

    return samples
