"""Prompting and few-shot formatting integration tests with real tokenizers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest
from datasets import Dataset

from eh30pt.constants import SENTINEL_TOKEN
from eh30pt.datasets.discontinuation.classify import (
    DiscontClassifier,
    DiscontClassifierRow,
    load_discont_classifier,
)
from eh30pt.datasets.discontinuation.decide import (
    DeciderRow,
    DiscontDecider,
    load_discont_decider,
)
from eh30pt.datasets.multinerd import Multinerd, MultinerdRow
from eh30pt.datasets.oblique import Obl, OblRow
from eh30pt.types import Architecture, FewShot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from transformers import PreTrainedTokenizerFast


@pytest.fixture
def oblique_sample() -> OblRow:
    """Return one oblique sample row."""
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


@pytest.fixture
def oblique_data() -> Dataset:
    """Return two oblique rows."""
    return Dataset.from_list(
        [
            {"sentence": "Tere maailm", "removed": "Tere", "label": "vaba"},
            {"sentence": "Kass jookseb", "removed": "Kass", "label": "seotud"},
        ]
    )


@pytest.fixture
def nerd_data() -> Dataset:
    """Return two multinerd rows."""
    return Dataset.from_list(
        [
            {
                "tokens": ["New", "York", "is", "big"],
                "ner_tags": [5, 6, 0, 0],
                "lang": "en",
            },
            {"tokens": ["Tallinn", "is"], "ner_tags": [5, 0], "lang": "en"},
        ]
    )


@pytest.fixture
def shots() -> list[FewShot]:
    """Return two oblique few-shot examples."""
    return [
        FewShot(user="Lause: a\nFraas: b\nKategooria:", label="vaba"),
        FewShot(user="Lause: c\nFraas: d\nKategooria:", label="seotud"),
    ]


@pytest.fixture
def counted_rows() -> Callable[[list[int]], Iterator[DiscontClassifierRow]]:
    """Return rows while recording each fresh iteration."""

    def _rows(calls: list[int]) -> Iterator[DiscontClassifierRow]:
        calls.append(1)

        yield {"id": 1, "text": "katkestas ravi"}
        yield {"id": 2, "text": "lõpetas kuuri"}

    return _rows


def test_format_few_shot_legacy(
    oblique_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
    shots: list[FewShot],
) -> None:
    ds = Obl(oblique_data, bert_tokenizer, "decoder")

    assert ds._format_few_shot_legacy([]) == ""
    assert ds._format_few_shot_legacy(shots) == (
        "Lause: a\nFraas: b\nKategooria: vaba\n"
        "\nLause: c\nFraas: d\nKategooria: seotud\n"
    )


def test_format_few_shot_chat_builds_turns(
    oblique_data: Dataset,
    llama_tokenizer: PreTrainedTokenizerFast,
    shots: list[FewShot],
) -> None:
    ds = Obl(oblique_data, llama_tokenizer, "decoder")

    assert ds._format_few_shot_chat(llama_tokenizer, []) == []
    assert ds._format_few_shot_chat(llama_tokenizer, shots) == [
        {"role": "user", "content": "Lause: a\nFraas: b\nKategooria:"},
        {"role": "assistant", "content": "vaba"},
        {"role": "user", "content": "Lause: c\nFraas: d\nKategooria:"},
        {"role": "assistant", "content": "seotud"},
    ]


def test_format_few_shot_chat_requires_template(
    oblique_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
    shots: list[FewShot],
) -> None:
    ds = Obl(oblique_data, bert_tokenizer, "decoder")

    with pytest.raises(RuntimeError, match="chat template"):
        ds._format_few_shot_chat(bert_tokenizer, shots)


def test_encode_prompt_legacy_round_trips(
    oblique_data: Dataset,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    shots: list[FewShot],
) -> None:
    ds = Obl(oblique_data, gpt2_tokenizer, "decoder")
    encoded = ds._encode_prompt(
        arch="decoder",
        tokenizer=gpt2_tokenizer,
        user_prompt="USER",
        few_shots=shots,
        max_len=None,
    )

    assert gpt2_tokenizer.decode(encoded["input_ids"]) == (
        f"{Obl.system_prompt}\nLause: a\nFraas: b\nKategooria: vaba\n"
        "\nLause: c\nFraas: d\nKategooria: seotud\n\nUSER"
    )


@pytest.mark.parametrize("arch", ["encoder", "decoder", "encoder-decoder"])
def test_encode_prompt_generation_flag(
    arch: Architecture,
    oblique_data: Dataset,
    gemma_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, gemma_tokenizer, arch)
    encoded = ds._encode_prompt(
        arch=arch,
        tokenizer=gemma_tokenizer,
        user_prompt="USER",
        few_shots=[],
        max_len=None,
    )
    decoded = gemma_tokenizer.decode(encoded["input_ids"])
    assert isinstance(decoded, str)

    assert "USER" in decoded
    if arch == "encoder":
        assert "<start_of_turn>model" not in decoded
    else:
        assert decoded.rstrip().endswith("<start_of_turn>model")


def test_encode_answer_legacy_appends_label(
    oblique_data: Dataset,
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, gpt2_tokenizer, "decoder")
    encoded = ds._encode_answer(
        tokenizer=gpt2_tokenizer,
        user_prompt="USER",
        label="vaba",
        few_shots=[],
        max_len=None,
    )

    assert gpt2_tokenizer.decode(encoded["input_ids"]) == (
        f"{Obl.system_prompt}\n\nUSER vaba"
    )


def test_encode_answer_template(
    oblique_data: Dataset,
    gemma_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, gemma_tokenizer, "decoder")
    encoded = ds._encode_answer(
        tokenizer=gemma_tokenizer,
        user_prompt="USER",
        label="vaba",
        few_shots=[],
        max_len=None,
    )
    decoded = gemma_tokenizer.decode(encoded["input_ids"])
    assert isinstance(decoded, str)

    assert "USER" in decoded
    assert decoded.rstrip().endswith("vaba<end_of_turn>")


def test_tokenize_decoder_masks_prompt(
    oblique_data: Dataset,
    oblique_sample: OblRow,
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, gpt2_tokenizer, "decoder")
    encoded = ds.tokenize(oblique_sample, gpt2_tokenizer)

    user = "Lause: Tere maailm\nFraas: Tere\nKategooria:"
    prompt_ids = ds._encode_prompt(
        arch="decoder",
        tokenizer=gpt2_tokenizer,
        user_prompt=user,
        few_shots=[],
        max_len=None,
    )["input_ids"]
    answer_ids = ds._encode_answer(
        tokenizer=gpt2_tokenizer,
        user_prompt=user,
        label="vaba",
        few_shots=[],
        max_len=None,
    )["input_ids"]

    assert encoded["labels"] == [-100] * len(prompt_ids) + answer_ids[len(prompt_ids) :]
    assert encoded["truncated"] is False


def test_tokenize_encoder_prompt_and_int_label(
    oblique_data: Dataset,
    oblique_sample: OblRow,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, bert_tokenizer, "encoder")
    encoded = ds.tokenize(oblique_sample, bert_tokenizer)

    assert encoded["labels"] == Obl.label2id["vaba"]
    assert encoded["truncated"] is False
    decoded = bert_tokenizer.decode(encoded["input_ids"])
    assert "tere" in decoded
    assert "[SEP]" in decoded


def test_tokenize_encoder_decoder_suffix_labels(
    oblique_data: Dataset,
    oblique_sample: OblRow,
    t5_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, t5_tokenizer, "encoder-decoder")
    encoded = ds.tokenize(oblique_sample, t5_tokenizer)

    user = "Lause: Tere maailm\nFraas: Tere\nKategooria:"
    prompt_ids = ds._encode_prompt(
        arch="encoder-decoder",
        tokenizer=t5_tokenizer,
        user_prompt=user,
        few_shots=[],
        max_len=None,
    )["input_ids"]
    answer_ids = ds._encode_answer(
        tokenizer=t5_tokenizer,
        user_prompt=user,
        label="vaba",
        few_shots=[],
        max_len=None,
    )["input_ids"]

    assert t5_tokenizer.eos_token_id is not None
    idx = len(prompt_ids) - int(answer_ids[-1] == t5_tokenizer.eos_token_id)

    assert encoded["labels"] == answer_ids[idx:]
    assert all(label != SENTINEL_TOKEN for label in encoded["labels"])
    assert encoded["truncated"] is False


def test_tokenize_multinerd_encoder_int_label(
    nerd_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Multinerd(nerd_data, bert_tokenizer, "encoder")
    row: MultinerdRow = {
        "sentence": "New York is big",
        "entity": "New York",
        "label": "LOC",
    }
    encoded = ds.tokenize(row, bert_tokenizer)

    assert encoded["labels"] == Multinerd.label2id["LOC"]
    decoded = bert_tokenizer.decode(encoded["input_ids"])
    assert isinstance(decoded, str)
    assert "new york" in decoded.lower()


def test_tokenize_over_request(
    oblique_data: Dataset,
    oblique_sample: OblRow,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, bert_tokenizer, "decoder")

    with pytest.raises(ValueError, match="more few shot examples than defined"):
        ds.tokenize(oblique_sample, bert_tokenizer, n_shot=99)


def test_samples_oblique(
    oblique_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, bert_tokenizer, "decoder")

    assert list(ds.samples()) == [
        FewShot(
            user="Lause: Tere maailm\nFraas: Tere\nKategooria:",
            label="vaba",
            id=0,
        ),
        FewShot(
            user="Lause: Kass jookseb\nFraas: Kass\nKategooria:",
            label="seotud",
            id=1,
        ),
    ]


def test_samples_multinerd_expands_entities(
    nerd_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Multinerd(nerd_data, bert_tokenizer, "decoder")

    assert list(ds.samples()) == [
        FewShot(
            user="Sentence: New York is big\nEntity: New York\nTag:",
            label="LOC",
            id=0,
        ),
        FewShot(
            user="Sentence: Tallinn is\nEntity: Tallinn\nTag:",
            label="LOC",
            id=1,
        ),
    ]


def test_set_epoch_ignored(
    oblique_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, bert_tokenizer, "decoder")

    assert ds.set_epoch(3) is None


def test_getitem_forbidden(
    oblique_data: Dataset,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ds = Obl(oblique_data, bert_tokenizer, "decoder")

    with pytest.raises(TypeError, match="indexing"):
        ds[0]


def test_classifier_streams_rows_lazily(
    counted_rows: Callable[[list[int]], Iterator[DiscontClassifierRow]],
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    calls: list[int] = []
    ds = DiscontClassifier(lambda: counted_rows(calls), bert_tokenizer, "decoder")

    assert calls == []

    first = list(ds.samples())

    assert calls == [1]
    assert first == [
        FewShot(user="katkestas ravi", label="", id=1),
        FewShot(user="lõpetas kuuri", label="", id=2),
    ]

    second = list(ds.samples())

    assert calls == [1, 1]
    assert [sample.user for sample in second] == ["katkestas ravi", "lõpetas kuuri"]


def test_decider_streams_rows_lazily(
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    calls: list[int] = []

    def _rows() -> Iterator[DeciderRow]:
        calls.append(1)

        yield {"id": 1, "text": "katkestas ravi"}
        yield {"id": 2, "text": "lõpetas kuuri"}

    ds = DiscontDecider(_rows, bert_tokenizer, "decoder")

    assert calls == []

    first = list(ds.samples())

    assert calls == [1]
    assert first == [
        FewShot(user="katkestas ravi", label="", id=1),
        FewShot(user="lõpetas kuuri", label="", id=2),
    ]

    second = list(ds.samples())

    assert calls == [1, 1]
    assert [sample.user for sample in second] == ["katkestas ravi", "lõpetas kuuri"]


def test_load_discont_decider_opens_no_connection(
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    splits = load_discont_decider(bert_tokenizer, "decoder")

    assert list(splits.train.samples()) == []
    assert list(splits.eval.samples()) == []


def test_load_classifier_opens_no_connection(
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    splits = load_discont_classifier(bert_tokenizer, "decoder")

    assert list(splits.train.samples()) == []
    assert list(splits.eval.samples()) == []
