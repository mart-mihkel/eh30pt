"""Pretrained model loading and param freezing."""

from typing import TYPE_CHECKING, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)

from eh30pt.logging import logger

if TYPE_CHECKING:
    from torch.nn import Module
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

    from eh30pt.types import Architecture, DatasetProtocol


def load_model(
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    data_info: DatasetProtocol,
    arch: Architecture,
    *,
    head_only: bool,
) -> PreTrainedModel:
    """Load a pretrained model for the architecture, optionally freezing the base."""
    have_cuda = torch.cuda.is_available()
    # deserializing bf16 weights directly onto MPS crashes, so load in fp32
    # on MPS and convert below
    have_mps = not have_cuda and torch.mps.is_available()
    if have_cuda:
        dtype = torch.bfloat16
        logger.debug("using bf16")
    else:
        dtype = torch.float32
        logger.debug("using full precision")

    skip_freeze = None
    if arch == "encoder":
        logger.debug("loading '%s' for sequence classification", model_path)
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            output_loading_info=True,
            ignore_mismatched_sizes=True,
            num_labels=len(data_info.id2label),
            id2label=data_info.id2label,
            label2id=data_info.label2id,
            device_map="auto",
            dtype=dtype,
        )

        skip_freeze = loading_info["missing_keys"]
    elif arch == "decoder":
        logger.debug("loading '%s' for causal language modeling", model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=dtype,
        )
    elif arch == "encoder-decoder":
        logger.debug("loading '%s' for sequence to sequence", model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=dtype,
        )

    if have_mps:
        logger.debug("converting to bf16")
        cast("Module", model).to(torch.bfloat16)

    config = model.config
    if "text_config" in config:
        logger.debug("using text config of multimodal model")
        config = config.text_config

    if config.pad_token_id is None:
        logger.warning("model doesn't have a padding token, using eos")
        config.pad_token_id = tokenizer.eos_token_id

    if head_only:
        _freeze(model, skip_freeze)

    return model


def _freeze(model: Module, skip: set[str] | None = None) -> None:
    """Freeze all model parameters except those named in `skip`."""
    if skip is None:
        skip = set()

    logger.info("freeze base model")
    for name, param in model.named_parameters():
        if name in skip:
            logger.info("skip '%s'", name)
            continue

        param.requires_grad = False
