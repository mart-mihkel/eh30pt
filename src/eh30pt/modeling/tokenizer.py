"""Tokenizer loading utilities."""

from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from eh30pt.logging import logger


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
