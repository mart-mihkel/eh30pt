"""QLoRA model construction: 4-bit quantization and LoRA adapter wrapping."""

from typing import TYPE_CHECKING, TypedDict, cast

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


class LoraKwargs(TypedDict, total=False):
    """Arch-specific extras for LoraConfig."""

    target_modules: str | list[str]
    target_parameters: list[str]


def _medgemma_lora_kwargs() -> LoraKwargs:
    """Return LoRA targets for MedGemma's text decoder, excluding its vision tower."""
    logger.debug("get lora args for medgemma")

    return {
        "target_modules": (
            r".*language_model\..*\."
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        ),
    }


def _mmbert_lora_kwargs() -> LoraKwargs:
    """Return LoRA targets for mmBERT's ModernBERT attention and MLP linears."""
    logger.debug("get lora args for mmbert")

    return {
        "target_modules": ["Wqkv", "Wi", "Wo"],
    }


def _qwen35_lora_kwargs() -> LoraKwargs:
    """Return LoRA targets for Qwen3.5's attention and MLP linears."""
    logger.debug("get lora args for qwen3.5")

    return {
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
        ],
    }


def _lora_kwargs(model_path: str) -> LoraKwargs:
    """Return model-specific LoRA extras."""
    path = model_path.lower()
    if "medgemma" in path:
        return _medgemma_lora_kwargs()

    if "mmbert" in path:
        return _mmbert_lora_kwargs()

    if "qwen3.5" in path or "qwen3_5" in path:
        return _qwen35_lora_kwargs()

    return {}


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
    *,
    sequence_classification: bool = False,
    lora_kwargs: LoraKwargs | None = None,
) -> LoraConfig:
    """Build the LoRA adapter config, using model-specific targets when provided."""
    target_modules = lora_kwargs.get("target_modules") if lora_kwargs else None
    target_parameters = lora_kwargs.get("target_parameters") if lora_kwargs else None

    task = get_task_type(
        arch,
        sequence_classification=sequence_classification,
    )

    return LoraConfig(
        task_type=task,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        target_parameters=target_parameters,
    )


def prepare_lora_model(
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    arch: Architecture,
    dataset: DatasetName,
    *,
    lora: LoRASettings,
    sequence_classification: bool = False,
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
        sequence_classification=sequence_classification,
        quantization_config=build_quantization_config(),
    )

    base = prepare_model_for_kbit_training(base)
    config = build_lora_config(
        lora.rank,
        lora.alpha,
        lora.dropout,
        arch,
        sequence_classification=sequence_classification,
        lora_kwargs=_lora_kwargs(model_path),
    )

    return cast("PeftModel", get_peft_model(base, config))
