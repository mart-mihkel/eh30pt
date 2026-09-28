"""Prompt-tuning construction integration tests with real tokenizers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from eh30pt.modeling.tuning import (
    _distiblert_prompt_tuning_kwargs,
    _t5gemma2_prompt_tuning_kwargs,
    get_n_virtual,
    prepare_pt_model,
)

if TYPE_CHECKING:
    import pytest

if TYPE_CHECKING:
    from peft import PromptTuningConfig
    from transformers import (
        DistilBertModel,
        PreTrainedTokenizerFast,
        T5Gemma2Model,
    )


def test_get_n_virtual_without_chat_template(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    prompt = "Classify the reason."
    expected = len(gpt2_tokenizer(prompt, truncation=True)["input_ids"])

    assert get_n_virtual(gpt2_tokenizer, prompt) == expected


def test_get_n_virtual_with_chat_template(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    gpt2_tokenizer.chat_template = (
        "{% for message in messages %}{{ message.content }}{% endfor %}"
    )
    try:
        prompt = "Classify the reason."
        conv = [{"role": "system", "content": prompt}]
        expected = len(
            cast(
                "dict[str, list[int]]",
                gpt2_tokenizer.apply_chat_template(conv, truncation=True),
            )["input_ids"]
        )

        assert get_n_virtual(gpt2_tokenizer, prompt) == expected
    finally:
        gpt2_tokenizer.chat_template = None


def test_distilbert_kwargs_read_config() -> None:
    model = SimpleNamespace(config=SimpleNamespace(dim=8, n_layers=2, n_heads=2))

    assert _distiblert_prompt_tuning_kwargs(cast("DistilBertModel", model)) == {
        "token_dim": 8,
        "num_layers": 2,
        "num_attention_heads": 2,
    }


def test_t5gemma2_kwargs_read_nested_config() -> None:
    text_config = SimpleNamespace(
        hidden_size=16, num_hidden_layers=3, num_attention_heads=4
    )
    encoder = SimpleNamespace(text_config=text_config)
    model = SimpleNamespace(config=SimpleNamespace(encoder=encoder))

    assert _t5gemma2_prompt_tuning_kwargs(cast("T5Gemma2Model", model)) == {
        "token_dim": 16,
        "num_layers": 3,
        "num_attention_heads": 4,
    }


def test_prepare_pt_model_wraps_gpt2(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    from test.conftest import models

    model = prepare_pt_model(
        "random",
        gpt2_tokenizer,
        models["gpt2"],
        "decoder",
        "oblique",
    )

    config = cast("PromptTuningConfig", model.peft_config["default"])

    assert config.num_virtual_tokens > 0


def test_prepare_pt_model_special_arch_branches(
    gpt2_tokenizer: PreTrainedTokenizerFast,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eh30pt.modeling import tuning

    distil = SimpleNamespace(
        config=SimpleNamespace(dim=8, n_layers=2, n_heads=2),
        peft_config={},
    )
    monkeypatch.setattr(tuning, "load_model", lambda *_args, **_kwargs: distil)
    monkeypatch.setattr(tuning, "get_peft_model", lambda base, _config: base)

    out = prepare_pt_model(
        "pretrained",
        gpt2_tokenizer,
        "some-distilbert-model",
        "encoder",
        "oblique",
    )

    assert out is distil

    text_config = SimpleNamespace(
        hidden_size=16, num_hidden_layers=3, num_attention_heads=4
    )
    t5 = SimpleNamespace(
        config=SimpleNamespace(encoder=SimpleNamespace(text_config=text_config)),
        peft_config={},
    )
    monkeypatch.setattr(tuning, "load_model", lambda *_args, **_kwargs: t5)

    out = prepare_pt_model(
        "random",
        gpt2_tokenizer,
        "some-t5gemma-2-model",
        "encoder-decoder",
        "oblique",
    )

    assert out is t5
