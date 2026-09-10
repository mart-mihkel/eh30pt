"""Architecture family inference from a model config."""

from typing import TYPE_CHECKING, cast

from peft import TaskType
from transformers import AutoModelForMaskedLM, AutoModelForSeq2SeqLM

from eh30pt.logging import logger

if TYPE_CHECKING:
    from transformers import PreTrainedConfig
    from transformers.models.auto.auto_factory import (
        _BaseAutoModelClass,
        _LazyAutoMapping,
    )

    from eh30pt.types import Architecture, PipelineTask


def _registered_for(cls: type, auto_cls: type[_BaseAutoModelClass]) -> bool:
    """Check whether `cls` is registered for `auto_cls`, transformers' only API here."""
    mapping = cast("_LazyAutoMapping", auto_cls._model_mapping)  # noqa: SLF001
    return cls in mapping


def infer_architecture(
    config: PreTrainedConfig,
    override: Architecture | None = None,
) -> Architecture:
    """Infer the model architecture family from its config."""
    if override is not None:
        logger.debug("using overridden architecture '%s'", override)
        return override

    cls = type(config)
    if config.is_encoder_decoder or _registered_for(cls, AutoModelForSeq2SeqLM):
        arch: Architecture = "encoder-decoder"
    elif _registered_for(cls, AutoModelForMaskedLM):
        arch = "encoder"
    else:
        arch = "decoder"

    logger.debug("inferred model architecture '%s' for '%s'", arch, config.model_type)
    return arch


def get_pipeline_task(arch: Architecture) -> PipelineTask:
    """Map architecture to its HuggingFace pipeline task."""
    if arch == "encoder":
        return "text-classification"

    if arch in {"decoder", "encoder-decoder"}:
        return "text-generation"

    msg = f"invalid architecture {arch}"
    raise ValueError(msg)


def get_task_type(arch: Architecture) -> TaskType:
    """Map architecture to peft task."""
    if arch == "encoder":
        return TaskType.SEQ_CLS

    if arch == "decoder":
        return TaskType.CAUSAL_LM

    if arch == "encoder-decoder":
        return TaskType.SEQ_2_SEQ_LM

    msg = f"invalid architecture {arch}"
    raise ValueError(msg)
