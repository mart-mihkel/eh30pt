"""Prompt-tuning model construction: virtual token counting and PEFT wrapping."""

from typing import TYPE_CHECKING, Any, cast

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

    from eh30pt.types import Architecture, DatasetProtocol, PrefixInit


def _distiblert_prompt_tuning_kwargs(model: DistilBertModel) -> dict[str, Any]:
    logger.debug("get prompt tuning args for distilbert")

    cfg = model.config
    return {
        "token_dim": cfg.dim,
        "num_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
    }


def _t5gemma2_prompt_tuning_kwargs(model: T5Gemma2Model) -> dict[str, Any]:
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
    """Count the tokens a virtual prompt initialized from `sys_prompt` will occupy."""
    if tokenizer.chat_template is None:
        sys_enc = tokenizer(sys_prompt, truncation=True)
    else:
        conv = [{"role": "system", "content": sys_prompt}]
        sys_enc = tokenizer.apply_chat_template(conv, truncation=True)

    return len(cast("dict[str, list[int]]", sys_enc)["input_ids"])


def prepare_pt_model(
    prefix_init: PrefixInit,
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    arch: Architecture,
    data: DatasetProtocol,
) -> PeftModel:
    """Load a pretrained model wrapped for prompt tuning."""
    logger.debug("preparing peft model for '%s'", model_path)
    sys_prompt = data.system_prompt
    n_virtual = get_n_virtual(tokenizer, sys_prompt)
    base = load_model(tokenizer, model_path, data, arch, head_only=False)
    task_type = get_task_type(arch)

    if prefix_init == "pretrained":
        init = "TEXT"
    elif prefix_init == "random":
        init = "RANDOM"

    special_kwargs = {}
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
        prompt_tuning_init_text=sys_prompt,
        num_virtual_tokens=n_virtual,
        **special_kwargs,
    )

    return cast("PeftModel", get_peft_model(base, config))
