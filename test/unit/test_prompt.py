"""Tests for prompt, example, answer, and conversation formatting."""

from typing import TYPE_CHECKING, Final
from unittest.mock import MagicMock

import pytest
from datasets import Dataset, DatasetDict

from eh30pt.datasets.multinerd import Multinerd
from eh30pt.datasets.multinerd import _Example as _MultinerdExample
from eh30pt.datasets.oblique import Oblique
from eh30pt.datasets.oblique import _Example as _ObliqueExample
from eh30pt.datasets.oblique import _Sample as _ObliqueSample

if TYPE_CHECKING:
    from eh30pt.types import Architecture

_max_length: Final = 512
_separator: Final = "[SEP]"


def _mock_tokenizer(chat_template: str | None, sep: str = _separator) -> MagicMock:
    return MagicMock(chat_template=chat_template, sep_token=sep)


def _oblique_sample() -> _ObliqueSample:
    return {
        "id": 1,
        "fpath": "f.txt",
        "sentence": "Tere maailm",
        "remove_start": 0,
        "remove_end": 4,
        "removed": "Tere",
        "label": "vaba",
        "short_sent": "Tere",
        "cons_score": 0.0,
        "ual": 0.0,
        "la": 0.0,
    }


def _oblique_row(sentence: str, removed: str, label: str) -> dict[str, str]:
    return {"sentence": sentence, "removed": removed, "label": label}


def _oblique_examples() -> list[_ObliqueExample]:
    return [
        _ObliqueExample(user="Lause: a\nFraas: b\nKategooria:", assistant="vaba"),
        _ObliqueExample(user="Lause: c\nFraas: d\nKategooria:", assistant="seotud"),
    ]


def test_oblique_user_prompt_decoder() -> None:
    ds = Oblique(_mock_tokenizer(None), "decoder")

    assert (
        ds._build_user_prompt(_oblique_sample())
        == "Lause: Tere maailm\nFraas: Tere\nKategooria:"
    )


def test_oblique_user_prompt_encoder_decoder() -> None:
    ds = Oblique(_mock_tokenizer(None), "encoder-decoder")

    assert (
        ds._build_user_prompt(_oblique_sample())
        == "lause: Tere maailm\nfraas: Tere\nkategooria:"
    )


def test_oblique_user_prompt_encoder_uses_sep_token() -> None:
    ds = Oblique(_mock_tokenizer(None, "|"), "encoder")

    assert ds._build_user_prompt(_oblique_sample()) == "Tere|Tere maailm"


def test_multinerd_user_prompt_decoder() -> None:
    ds = Multinerd(_mock_tokenizer(None), "decoder")

    assert (
        ds._build_user_prompt("Cats run fast", "Cats")
        == "Sentence: Cats run fast\nEntity: Cats\nTag:"
    )


def test_multinerd_user_prompt_encoder_uses_sep_token() -> None:
    ds = Multinerd(_mock_tokenizer(None, "|"), "encoder")

    assert ds._build_user_prompt("Cats run fast", "Cats") == "Cats run fast|Cats"


def test_oblique_format_examples_legacy_empty() -> None:
    ds = Oblique(_mock_tokenizer(None), "decoder")

    assert ds._format_examples_legacy([]) == ""


def test_oblique_format_examples_legacy_joins() -> None:
    ds = Oblique(_mock_tokenizer(None), "decoder")

    assert ds._format_examples_legacy(_oblique_examples()) == (
        "Lause: a\nFraas: b\nKategooria: vaba\n"
        "\nLause: c\nFraas: d\nKategooria: seotud\n"
    )


def test_multinerd_format_examples_legacy_joins() -> None:
    ds = Multinerd(_mock_tokenizer(None), "decoder")
    examples = [
        _MultinerdExample(user="Sentence: s\nEntity: e\nTag:", assistant="PER"),
    ]

    assert ds._format_examples_legacy(examples) == "Sentence: s\nEntity: e\nTag: PER\n"


def test_oblique_format_examples_template_requires_chat_template() -> None:
    ds = Oblique(_mock_tokenizer(None), "decoder")

    with pytest.raises(RuntimeError, match="chat template"):
        ds._format_examples_template(_oblique_examples())


def test_multinerd_format_examples_template_requires_chat_template() -> None:
    ds = Multinerd(_mock_tokenizer(None), "decoder")
    examples = [_MultinerdExample(user="q", assistant="PER")]

    with pytest.raises(RuntimeError, match="chat template"):
        ds._format_examples_template(examples)


def test_oblique_format_examples_template_builds_turns() -> None:
    ds = Oblique(_mock_tokenizer("dummy"), "decoder")

    assert ds._format_examples_template([]) == []
    assert ds._format_examples_template(_oblique_examples()) == [
        {"role": "user", "content": "Lause: a\nFraas: b\nKategooria:"},
        {"role": "assistant", "content": "vaba"},
        {"role": "user", "content": "Lause: c\nFraas: d\nKategooria:"},
        {"role": "assistant", "content": "seotud"},
    ]


def test_oblique_encode_prompt_legacy() -> None:
    tokenizer = _mock_tokenizer(None)
    ds = Oblique(tokenizer, "decoder")
    ds._encode_prompt("SYS", "USER", None, _oblique_examples())

    tokenizer.assert_called_once_with(
        "SYS\nLause: a\nFraas: b\nKategooria: vaba\n"
        "\nLause: c\nFraas: d\nKategooria: seotud\n\nUSER",
        truncation=True,
        max_length=None,
    )


def test_multinerd_encode_prompt_legacy_empty_examples() -> None:
    tokenizer = _mock_tokenizer(None)
    ds = Multinerd(tokenizer, "decoder")
    ds._encode_prompt("SYS", "USER", None, [])

    tokenizer.assert_called_once_with("SYS\n\nUSER", truncation=True, max_length=None)


def test_oblique_encode_prompt_template_conversation() -> None:
    tokenizer = _mock_tokenizer("dummy")
    ds = Oblique(tokenizer, "decoder")
    ds._encode_prompt("SYS", "USER", _max_length, _oblique_examples())

    call = tokenizer.apply_chat_template.call_args
    assert call is not None
    assert call.args[0] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Lause: a\nFraas: b\nKategooria:"},
        {"role": "assistant", "content": "vaba"},
        {"role": "user", "content": "Lause: c\nFraas: d\nKategooria:"},
        {"role": "assistant", "content": "seotud"},
        {"role": "user", "content": "USER"},
    ]


@pytest.mark.parametrize("arch", ["encoder", "decoder", "encoder-decoder"])
def test_oblique_encode_prompt_generation_flag(arch: Architecture) -> None:
    tokenizer = _mock_tokenizer("dummy")
    ds = Oblique(tokenizer, arch)
    ds._encode_prompt("SYS", "USER", None, [])

    call = tokenizer.apply_chat_template.call_args
    assert call is not None
    assert [m["role"] for m in call.args[0]] == ["system", "user"]


@pytest.mark.parametrize("arch", ["encoder", "decoder", "encoder-decoder"])
def test_multinerd_encode_prompt_generation_flag(arch: Architecture) -> None:
    tokenizer = _mock_tokenizer("dummy")
    ds = Multinerd(tokenizer, arch)
    ds._encode_prompt("SYS", "USER", None, [])

    call = tokenizer.apply_chat_template.call_args
    assert call is not None
    assert [m["role"] for m in call.args[0]] == ["system", "user"]


def test_oblique_encode_answer_legacy() -> None:
    tokenizer = _mock_tokenizer(None)
    ds = Oblique(tokenizer, "decoder")
    ds._encode_answer("SYS", "USER", "vaba", None, _oblique_examples())

    tokenizer.assert_called_once_with(
        "SYS\nLause: a\nFraas: b\nKategooria: vaba\n"
        "\nLause: c\nFraas: d\nKategooria: seotud\n\nUSER vaba",
        truncation=True,
        max_length=None,
    )


def test_oblique_encode_answer_template() -> None:
    tokenizer = _mock_tokenizer("dummy")
    ds = Oblique(tokenizer, "decoder")
    ds._encode_answer("SYS", "USER", "vaba", None, _oblique_examples())

    call = tokenizer.apply_chat_template.call_args
    assert call is not None
    assert call.args[0][-2:] == [
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": "vaba"},
    ]


def test_multinerd_encode_answer_legacy() -> None:
    tokenizer = _mock_tokenizer(None)
    ds = Multinerd(tokenizer, "decoder")
    ds._encode_answer("SYS", "USER", 0, None, [])

    tokenizer.assert_called_once_with(
        "SYS\n\nUSER PER", truncation=True, max_length=None
    )


def test_multinerd_encode_answer_template() -> None:
    tokenizer = _mock_tokenizer("dummy")
    ds = Multinerd(tokenizer, "decoder")
    ds._encode_answer("SYS", "USER", 3, None, [])

    call = tokenizer.apply_chat_template.call_args
    assert call is not None
    assert call.args[0][-2:] == [
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": "ANIM"},
    ]


def test_multinerd_sample_examples_joins_spans() -> None:
    ds = Multinerd(_mock_tokenizer(None), "decoder")
    ds._raw_data = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {
                        "tokens": ["New", "York", "is", "big"],
                        "ner_tags": [5, 6, 0, 0],
                        "lang": "en",
                    }
                ]
            )
        }
    )

    assert ds._sample_examples(1) == [
        _MultinerdExample(
            user="Sentence: New York is big\nEntity: New York\nTag:",
            assistant="LOC",
        )
    ]


def test_multinerd_sample_examples_over_request() -> None:
    ds = Multinerd(_mock_tokenizer(None), "decoder")
    ds._raw_data = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {
                        "tokens": ["New", "York"],
                        "ner_tags": [5, 6],
                        "lang": "en",
                    }
                ]
            )
        }
    )

    with pytest.raises(ValueError, match="requested more than 1 examples"):
        ds._sample_examples(2)


def test_oblique_sample_examples_selects_first_rows() -> None:
    ds = Oblique(_mock_tokenizer(None), "decoder")
    ds._raw_data = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    _oblique_row("Lause üks", "üks", "vaba"),
                    _oblique_row("Lause kaks", "kaks", "seotud"),
                    _oblique_row("Lause kolm", "kolm", "vaba"),
                ]
            )
        }
    )

    examples = ds._sample_examples(2)

    assert examples == [
        _ObliqueExample(
            user="Lause: Lause üks\nFraas: üks\nKategooria:",
            assistant="vaba",
        ),
        _ObliqueExample(
            user="Lause: Lause kaks\nFraas: kaks\nKategooria:",
            assistant="seotud",
        ),
    ]


def test_oblique_sample_examples_over_request() -> None:
    ds = Oblique(_mock_tokenizer(None), "decoder")
    ds._raw_data = DatasetDict(
        {"train": Dataset.from_list([_oblique_row("Lause üks", "üks", "vaba")])}
    )

    with pytest.raises(ValueError, match="requested more than 1 examples"):
        ds._sample_examples(2)
