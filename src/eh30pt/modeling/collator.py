"""Batch collation with HuggingFace data collators."""

from typing import TYPE_CHECKING

from transformers import (
    DataCollator,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
)

from eh30pt.constants import PAD_MULTIPLE
from eh30pt.logging import logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from eh30pt.types import Architecture


def get_collator(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    *,
    sequence_classification: bool = False,
) -> DataCollator:
    """Build the padding collator appropriate for the given architecture."""
    logger.debug("init data collator for '%s'", arch)
    if sequence_classification or arch == "encoder":
        return DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=PAD_MULTIPLE,
        )

    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=PAD_MULTIPLE,
    )
