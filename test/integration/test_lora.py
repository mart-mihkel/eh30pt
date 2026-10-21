"""QLoRA construction integration tests with real tiny models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from peft import PeftModel

from eh30pt.modeling.lora import (
    _lora_kwargs,
    build_lora_config,
    prepare_lora_model,
)
from eh30pt.settings import LoRASettings

if TYPE_CHECKING:
    from peft import LoraConfig
    from transformers import PreTrainedTokenizerFast

_lora_r: Final = 16


def test_prepare_lora_model_wraps_quantized_base(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    model = prepare_lora_model(
        gpt2_tokenizer,
        "hf-internal-testing/tiny-random-gpt2",
        "decoder",
        "oblique",
        lora=LoRASettings(rank=16, alpha=32, dropout=0.05),
    )

    assert isinstance(model, PeftModel)
    assert cast("LoraConfig", model.peft_config["default"]).r == _lora_r


def test_medgemma_lora_adapters_cover_only_the_text_language_model() -> None:
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-Gemma3ForConditionalGeneration",
    )
    config = build_lora_config(
        _lora_r,
        32,
        0.05,
        "decoder",
        lora_kwargs=_lora_kwargs("google/medgemma-4b-it"),
    )
    wrapped = get_peft_model(base, config)
    adapted = [
        name
        for name, _module in wrapped.named_modules()
        if name.endswith("lora_A.default")
    ]

    assert adapted
    assert all("language_model" in name for name in adapted)
    assert not any("vision_tower" in name for name in adapted)


def test_mmbert_lora_adapters_cover_modernbert_linears() -> None:
    from peft import get_peft_model
    from transformers import AutoModelForSequenceClassification

    base = AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-ModernBertForSequenceClassification",
    )
    config = build_lora_config(
        _lora_r,
        32,
        0.05,
        "encoder",
        lora_kwargs=_lora_kwargs("jhu-clsp/mmBERT-base"),
    )
    wrapped = get_peft_model(base, config)
    adapted = [
        name
        for name, _module in wrapped.named_modules()
        if name.endswith("lora_A.default")
    ]
    targets = {
        name.removesuffix(".lora_A.default").rsplit(".", 1)[-1] for name in adapted
    }

    assert adapted
    assert targets == {"Wqkv", "Wi", "Wo"}
    assert not any(".head." in name for name in adapted)
