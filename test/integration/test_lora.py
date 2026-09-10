"""QLoRA construction integration test with a real quantized tiny model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from peft import PeftModel

from eh30pt.modeling.lora import prepare_lora_model
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
