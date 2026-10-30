"""Pretrained model loading and param freezing."""

from typing import TYPE_CHECKING, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)

from eh30pt.datasets.registry import assert_classification, get_labels
from eh30pt.logging import logger

if TYPE_CHECKING:
    from torch.nn import Module
    from transformers import (
        BitsAndBytesConfig,
        PreTrainedModel,
        PreTrainedTokenizerFast,
    )

    from eh30pt.types import Architecture, DatasetName


def load_model(
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    dataset: DatasetName,
    arch: Architecture,
    *,
    head_only: bool = False,
    sequence_classification: bool = False,
    quantization_config: BitsAndBytesConfig | None = None,
) -> PreTrainedModel:
    """Load a pretrained model for the architecture, optionally freezing the base."""
    assert_classification(dataset, "model heads need labels")
    sequence_classification = sequence_classification or arch == "encoder"

    logger.info("loading '%s'", model_path)
    have_cuda = torch.cuda.is_available()
    have_mps = not have_cuda and torch.mps.is_available()
    if have_cuda:
        dtype = torch.bfloat16
        device_map = "auto"
        logger.debug("using bf16")
    elif have_mps:
        dtype = torch.bfloat16
        device_map = "cpu"
        logger.debug("using bf16")
    else:
        dtype = torch.float32
        device_map = "auto"
        logger.debug("using full precision")

    attn = _attention_implementation(model_path)
    skip_freeze = None
    if sequence_classification:
        labels = get_labels(dataset)
        id2label = dict(enumerate(labels))
        label2id = {label: index for index, label in id2label.items()}
        logger.debug("loading model for sequence classification")
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            output_loading_info=True,
            ignore_mismatched_sizes=True,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            device_map=device_map,
            dtype=dtype,
            quantization_config=quantization_config,
            attn_implementation=attn,
        )

        skip_freeze = loading_info["missing_keys"]
    elif arch == "decoder":
        logger.debug("loading model for causal language modeling")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            dtype=dtype,
            quantization_config=quantization_config,
            attn_implementation=attn,
        )
    elif arch == "encoder-decoder":
        logger.debug("loading model for sequence to sequence", model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            device_map=device_map,
            dtype=dtype,
            quantization_config=quantization_config,
            attn_implementation=attn,
        )

    if have_mps:
        logger.debug("moving model to mps")
        cast("Module", model).to("mps")

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


def _attention_implementation(model_path: str) -> str | None:
    """Use eager attention for MedGemma and the Transformers default otherwise."""
    return "eager" if "medgemma" in model_path.lower() else None


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
