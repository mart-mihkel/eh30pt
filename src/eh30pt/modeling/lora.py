"""QLoRA model construction: 4-bit quantization and LoRA adapter wrapping."""

from typing import TYPE_CHECKING, cast

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from eh30pt.logging import logger
from eh30pt.modeling.arch import get_task_type
from eh30pt.modeling.loader import load_model

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedTokenizerFast

    from eh30pt.settings import LoRASettings
    from eh30pt.types import Architecture, DatasetName


def build_quantization_config() -> BitsAndBytesConfig:
    """Build the double-quantized 4-bit NF4 config used for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    arch: Architecture,
) -> LoraConfig:
    """Build the LoRA adapter config for architecture, inferring target modules."""
    return LoraConfig(
        task_type=get_task_type(arch),
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )


def prepare_lora_model(
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    arch: Architecture,
    dataset: DatasetName,
    *,
    lora: LoRASettings,
) -> PeftModel:
    """Load a 4-bit quantized model wrapped with a trainable LoRA adapter."""
    logger.info(
        "preparing qlora model for '%s' with rank %d, alpha %d, dropout %.2f",
        model_path,
        lora.rank,
        lora.alpha,
        lora.dropout,
    )

    base = load_model(
        tokenizer,
        model_path,
        dataset,
        arch,
        head_only=False,
        quantization_config=build_quantization_config(),
    )

    base = prepare_model_for_kbit_training(base)
    config = build_lora_config(lora.rank, lora.alpha, lora.dropout, arch)

    return cast("PeftModel", get_peft_model(base, config))
