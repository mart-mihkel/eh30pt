"""Prompt-tuning construction integration tests with real tokenizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from peft import PeftModel
from transformers import (
    DistilBertConfig,
    DistilBertModel,
    T5Gemma2Config,
    T5Gemma2DecoderConfig,
    T5Gemma2EncoderConfig,
    T5Gemma2ForConditionalGeneration,
    T5Gemma2TextConfig,
)

from eh30pt.datasets.registry import get_sys_prompt
from eh30pt.modeling import tuning
from eh30pt.modeling.tuning import (
    _distiblert_prompt_tuning_kwargs,
    _t5gemma2_prompt_tuning_kwargs,
    get_n_virtual,
    prepare_pt_model,
)
from test.conftest import models

if TYPE_CHECKING:
    import pytest
    from peft import PromptTuningConfig
    from transformers import PreTrainedTokenizerFast


def test_get_n_virtual_without_chat_template(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    prompt = get_sys_prompt("oblique")
    expected = len(gpt2_tokenizer(prompt, truncation=True)["input_ids"])

    assert get_n_virtual(gpt2_tokenizer, prompt) == expected


def test_get_n_virtual_ignores_chat_template(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    gpt2_tokenizer.chat_template = "{{ 'x' }}"

    try:
        prompt = get_sys_prompt("oblique")
        expected = len(gpt2_tokenizer(prompt, truncation=True)["input_ids"])
        assert expected > 1
        assert get_n_virtual(gpt2_tokenizer, prompt) == expected
    finally:
        gpt2_tokenizer.chat_template = None


def test_distilbert_kwargs_read_config() -> None:
    config = DistilBertConfig(dim=32, n_layers=2, n_heads=4, vocab_size=128)
    model = DistilBertModel(config)

    assert _distiblert_prompt_tuning_kwargs(model) == {
        "token_dim": 32,
        "num_layers": 2,
        "num_attention_heads": 4,
    }


def test_t5gemma2_kwargs_read_nested_config() -> None:
    from transformers import (
        T5Gemma2Config,
        T5Gemma2DecoderConfig,
        T5Gemma2EncoderConfig,
        T5Gemma2Model,
        T5Gemma2TextConfig,
    )

    text = T5Gemma2TextConfig(
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=32,
        vocab_size=128,
    )

    encoder = T5Gemma2EncoderConfig(text_config=text)
    decoder = T5Gemma2DecoderConfig(
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=32,
        vocab_size=128,
    )

    model = T5Gemma2Model(T5Gemma2Config(encoder=encoder, decoder=decoder))

    assert _t5gemma2_prompt_tuning_kwargs(model) == {
        "token_dim": 16,
        "num_layers": 3,
        "num_attention_heads": 4,
    }


def test_prepare_pt_model_wraps_gpt2(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    model = prepare_pt_model(
        "random",
        gpt2_tokenizer,
        models["gpt2"],
        "decoder",
        "oblique",
        system_prompt=get_sys_prompt("oblique", "decoder"),
        num_virtual=get_n_virtual(gpt2_tokenizer, get_sys_prompt("oblique", "decoder")),
    )

    config = cast("PromptTuningConfig", model.peft_config["default"])

    assert config.num_virtual_tokens > 0


def test_prepare_pt_model_special_arch_branches(
    gpt2_tokenizer: PreTrainedTokenizerFast,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distil = DistilBertModel(
        DistilBertConfig(dim=32, n_layers=2, n_heads=4, vocab_size=128)
    )

    monkeypatch.setattr(tuning, "load_model", lambda *_args, **_kwargs: distil)

    out = prepare_pt_model(
        "random",
        gpt2_tokenizer,
        "some-distilbert-model",
        "encoder",
        "oblique",
        system_prompt=get_sys_prompt("oblique", "encoder"),
        num_virtual=get_n_virtual(gpt2_tokenizer, get_sys_prompt("oblique", "encoder")),
    )

    assert isinstance(out, PeftModel)
    assert cast("PromptTuningConfig", out.peft_config["default"]).token_dim == (
        distil.config.dim
    )

    text = T5Gemma2TextConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        vocab_size=128,
    )

    t5 = T5Gemma2ForConditionalGeneration(
        T5Gemma2Config(
            encoder=T5Gemma2EncoderConfig(text_config=text),
            decoder=T5Gemma2DecoderConfig(
                hidden_size=16,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=32,
                vocab_size=128,
            ),
        )
    )

    monkeypatch.setattr(tuning, "load_model", lambda *_args, **_kwargs: t5)

    out = prepare_pt_model(
        "random",
        gpt2_tokenizer,
        "some-t5gemma-2-model",
        "encoder-decoder",
        "oblique",
        system_prompt=get_sys_prompt("oblique", "encoder-decoder"),
        num_virtual=get_n_virtual(
            gpt2_tokenizer, get_sys_prompt("oblique", "encoder-decoder")
        ),
    )

    assert isinstance(out, PeftModel)
    encoder = cast("T5Gemma2EncoderConfig", t5.config.encoder)
    text_config = cast("T5Gemma2TextConfig", encoder.text_config)
    assert (
        cast("PromptTuningConfig", out.peft_config["default"]).token_dim
        == text_config.hidden_size
    )
