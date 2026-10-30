"""Prompt-tuning model construction: virtual token counting and PEFT wrapping."""

from typing import TYPE_CHECKING, TypedDict, cast

from peft import PeftModel, PromptTuningConfig, get_peft_model

from eh30pt.logging import logger
from eh30pt.modeling.arch import get_task_type
from eh30pt.modeling.loader import load_model

if TYPE_CHECKING:
    from transformers import (
        DistilBertModel,
        PreTrainedTokenizerFast,
        T5Gemma2EncoderConfig,
        T5Gemma2Model,
        T5Gemma2TextConfig,
    )

    from eh30pt.types import Architecture, DatasetName, PrefixInit


class PromptTuningKwargs(TypedDict, total=False):
    """Arch-specific extras for PromptTuningConfig."""

    token_dim: int
    num_layers: int
    num_attention_heads: int


def _distiblert_prompt_tuning_kwargs(model: DistilBertModel) -> PromptTuningKwargs:
    logger.debug("get prompt tuning args for distilbert")

    cfg = model.config
    return {
        "token_dim": cfg.dim,
        "num_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
    }


def _t5gemma2_prompt_tuning_kwargs(model: T5Gemma2Model) -> PromptTuningKwargs:
    logger.debug("get prompt tuning args for t5gemma-2")

    cfg = cast("T5Gemma2EncoderConfig", model.config.encoder)
    cfg = cast("T5Gemma2TextConfig", cfg.text_config)
    return {
        "token_dim": cfg.hidden_size,
        "num_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
    }


def get_n_virtual(
    tokenizer: PreTrainedTokenizerFast,
    sys_prompt: str,
) -> int:
    """Count the tokens PEFT will use to initialize the virtual prompt."""
    return len(tokenizer(sys_prompt, truncation=True)["input_ids"])


def prepare_pt_model(
    prefix_init: PrefixInit,
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    arch: Architecture,
    dataset: DatasetName,
    *,
    system_prompt: str,
    num_virtual: int,
    sequence_classification: bool = False,
) -> PeftModel:
    """Load a pretrained model wrapped for prompt tuning."""
    logger.info(
        "preparing prompt tuning model for '%s' with %s prefix initialization",
        model_path,
        prefix_init,
    )

    base = load_model(
        tokenizer,
        model_path,
        dataset,
        arch,
        sequence_classification=sequence_classification,
        head_only=False,
    )

    task_type = get_task_type(
        arch,
        sequence_classification=sequence_classification,
    )

    if prefix_init == "pretrained":
        init = "TEXT"
    elif prefix_init == "random":
        init = "RANDOM"

    special_kwargs: PromptTuningKwargs = {}
    if "distilbert" in model_path:
        base = cast("DistilBertModel", base)
        special_kwargs = _distiblert_prompt_tuning_kwargs(base)
    elif "t5gemma-2" in model_path:
        base = cast("T5Gemma2Model", base)
        special_kwargs = _t5gemma2_prompt_tuning_kwargs(base)

    config = PromptTuningConfig(
        task_type=task_type,
        prompt_tuning_init=init,
        tokenizer_name_or_path=model_path,
        prompt_tuning_init_text=system_prompt,
        num_virtual_tokens=num_virtual,
        **special_kwargs,
    )

    return cast("PeftModel", get_peft_model(base, config))
