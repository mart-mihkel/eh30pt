"""Tests for prompt-tuning virtual token counting."""

from typing import TYPE_CHECKING, cast

from eh30pt.modeling.tuning import get_n_virtual

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


def test_get_n_virtual_without_chat_template(
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    n = get_n_virtual(bert_tokenizer, "hello there")
    expected = len(bert_tokenizer("hello there")["input_ids"])

    assert n == expected


def test_get_n_virtual_with_chat_template(
    llama_tokenizer: PreTrainedTokenizerFast,
) -> None:
    n = get_n_virtual(llama_tokenizer, "hello there")
    conv = [{"role": "system", "content": "hello there"}]
    sys_enc = cast(
        "dict[str, list[int]]",
        llama_tokenizer.apply_chat_template(conv),
    )
    expected = len(sys_enc["input_ids"])

    assert n == expected
