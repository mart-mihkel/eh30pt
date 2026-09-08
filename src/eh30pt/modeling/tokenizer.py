"""Tokenizer loading utilities."""

from typing import TYPE_CHECKING, cast

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from eh30pt.logging import logger

if TYPE_CHECKING:
    from transformers import PreTrainedConfig


def load_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    """Load a tokenizer, falling back to the eos token for padding if needed."""
    logger.info("loading pretrained tokenizer for '%s'", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = cast("PreTrainedTokenizerFast", tokenizer)

    if tokenizer.pad_token is None:
        logger.warning("tokenizer doesn't have a padding token, using eos")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def cap_tokenizer_length(
    tokenizer: PreTrainedTokenizerFast,
    config: PreTrainedConfig,
) -> None:
    """
    Cap the tokenizer length at the model's position limit, if it has one.

    Tokenizers sometimes claim longer contexts than their model supports
    (or no limit at all), letting overlong sequences reach the model and
    fail in position embeddings. The model limit wins when present.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerFast
        Tokenizer whose length budget is capped in place.
    config : PreTrainedConfig
        Model config providing the position limit, if it defines one.

    """
    limit = cast("int | None", getattr(config, "n_positions", None))
    if limit is None:
        limit = cast("int | None", getattr(config, "max_position_embeddings", None))

    if limit is not None:
        tokenizer.model_max_length = min(tokenizer.model_max_length, limit)
