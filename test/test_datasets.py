"""Tests for dataset pipeline classes."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from eh30pt.constants import SENTINEL_TOKEN
from eh30pt.datasets.multinerd import Multinerd
from eh30pt.datasets.oblique import Oblique
from eh30pt.datasets.registry import get_dataset, get_sys_prompt, load_data
from eh30pt.modeling.collator import get_collator
from eh30pt.types import DatasetProtocol

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from eh30pt.datasets.oblique import _Example
    from eh30pt.types import DatasetName

_SEP = "[SEP]"


class _StubTokenizer:
    """Offline tokenizer double returning alternating short/long ids."""

    sep_token = _SEP
    chat_template = None
    model_max_length = 10**9
    eos_token_id = 99

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        del text, kwargs
        self.calls += 1
        ids = [7, 8] if self.calls % 2 == 1 else [7, 8, 9]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _tokenizer() -> PreTrainedTokenizerFast:
    return cast("PreTrainedTokenizerFast", SimpleNamespace(sep_token=_SEP))


def _stub_tokenizer() -> PreTrainedTokenizerFast:
    return cast("PreTrainedTokenizerFast", _StubTokenizer())


def _multinerd_frame() -> dict:
    return {
        "tokens": [["John", "works"], ["Acme", "hired", "her"]],
        "ner_tags": [[1, 0], [3, 0, 0]],
        "lang": ["en", "en"],
    }


def _obl_frame(n: int = 1200) -> dict:
    labels = ["bound", "free", "unnatural", "dubious", "redundant comma"]
    rows = [
        {
            "id": i,
            "fpath": f"f{i}",
            "sentence": f"lause {i}",
            "remove_start": 0,
            "remove_end": 1,
            "removed": f"fraas {i}",
            "type": labels[i % 5],
            "short_sent": f"l{i}",
            "cons_score": 1.0,
            "ual": 0.5,
            "la": 0.5,
        }
        for i in range(n)
    ]
    return {key: [row[key] for row in rows] for key in rows[0]}


def _obl_example() -> _Example:
    example: _Example = {
        "id": 0,
        "fpath": "f",
        "sentence": "Tere maailm",
        "remove_start": 5,
        "remove_end": 11,
        "removed": "maailm",
        "label": "seotud",
        "short_sent": "Tere",
        "cons_score": 1.0,
        "ual": 0.5,
        "la": 0.5,
    }
    return example


class _CsvStub:
    """Stub for `datasets.Dataset` reading the OBL csv from test rows."""

    @classmethod
    def from_csv(cls, *args: object, **kwargs: object) -> Dataset:
        del args, kwargs
        return Dataset.from_dict(_obl_frame())


def test_pipelines_satisfy_protocol() -> None:
    assert isinstance(Multinerd(_tokenizer(), "decoder"), DatasetProtocol)
    assert isinstance(Oblique(_tokenizer(), "decoder"), DatasetProtocol)


def test_label_maps_are_inverse() -> None:
    for label_id, label in Multinerd.id2label.items():
        assert Multinerd.label2id[label] == label_id
    for label_id, label in Oblique.id2label.items():
        assert Oblique.label2id[label] == label_id


def test_guards_reject_unloaded_state() -> None:
    for cls in (Multinerd, Oblique):
        dataset = cls(_tokenizer(), "decoder")
        with pytest.raises(RuntimeError, match="not loaded"):
            _ = dataset.data
        with pytest.raises(RuntimeError, match="not loaded"):
            _ = dataset.raw_data


def test_multinerd_system_prompts() -> None:
    encoder = Multinerd(_tokenizer(), "encoder").system_prompt
    decoder = Multinerd(_tokenizer(), "decoder").system_prompt
    encdec = Multinerd(_tokenizer(), "encoder-decoder").system_prompt

    assert encoder.endswith(_SEP)
    assert decoder == encdec
    assert "PER" in decoder
    assert "Output only the tag." in decoder


def test_oblique_system_prompts() -> None:
    encoder = Oblique(_tokenizer(), "encoder").system_prompt
    decoder = Oblique(_tokenizer(), "decoder").system_prompt
    encdec = Oblique(_tokenizer(), "encoder-decoder").system_prompt

    assert encoder.endswith(_SEP)
    assert "Väljasta ainult kategooria." in decoder
    assert "väljasta ainult kategooria" in encdec
    assert decoder != encdec


def test_multinerd_join_spans() -> None:
    tokens = ["Kuulus", "kohver", "Eston", "Kohver"]
    entities, tag_ids = Multinerd._join_spans(tokens=tokens, tag_ids=[0, 0, 1, 2])

    assert entities == ["Kuulus", "kohver", "Eston Kohver"]
    assert tag_ids == [-1, -1, 0]


def test_oblique_format_example() -> None:
    assert Oblique._format_example(_obl_example()) == (
        "Lause: Tere maailm\nFraas: maailm\nKategooria: seotud\n"
    )


def test_oblique_translate_en_to_et() -> None:
    example = cast("_Example", {**_obl_example(), "label": "bound"})

    assert Oblique._translate_en_to_et(example)["label"] == "seotud"


def test_multinerd_build_prompt() -> None:
    dataset = Multinerd(_tokenizer(), "decoder")

    assert dataset._build_prompt("s", "e", []) == "Sentence: s\nEntity: e\nTag:"
    assert dataset._build_prompt("s", "e", ["shot"]) == (
        "shot\nSentence: s\nEntity: e\nTag:"
    )


def test_oblique_build_prompt() -> None:
    dataset = Oblique(_tokenizer(), "decoder")

    assert dataset._build_prompt(_obl_example(), []) == (
        "Lause: Tere maailm\nFraas: maailm\nKategooria:"
    )


def test_multinerd_load_encoder() -> None:
    dataset = Multinerd(_stub_tokenizer(), "encoder")
    raw = DatasetDict({"train": Dataset.from_dict(_multinerd_frame())})
    with patch("eh30pt.datasets.multinerd.load_dataset", return_value=raw):
        assert dataset.load() is None

    assert set(dataset.raw_data["train"].column_names) == {
        "tokens",
        "ner_tags",
        "lang",
    }
    assert [row["labels"] for row in dataset.data["train"]] == [0, 1]
    assert "token_type_ids" not in dataset.data["train"].column_names


def test_multinerd_load_decoder_masks_prompt() -> None:
    dataset = Multinerd(_stub_tokenizer(), "decoder")
    raw = DatasetDict({"train": Dataset.from_dict(_multinerd_frame())})
    with patch("eh30pt.datasets.multinerd.load_dataset", return_value=raw):
        dataset.load()

    for row in dataset.data["train"]:
        assert len(row["labels"]) == len(row["input_ids"])
        assert row["labels"][-1] != SENTINEL_TOKEN


def test_multinerd_load_encoder_decoder() -> None:
    dataset = Multinerd(_stub_tokenizer(), "encoder-decoder")
    raw = DatasetDict({"train": Dataset.from_dict(_multinerd_frame())})
    with patch("eh30pt.datasets.multinerd.load_dataset", return_value=raw):
        dataset.load()

    for row in dataset.data["train"]:
        assert all(label >= 0 for label in row["labels"])


def test_multinerd_sample_examples() -> None:
    dataset = Multinerd(_stub_tokenizer(), "decoder")
    raw = DatasetDict({"train": Dataset.from_dict(_multinerd_frame())})
    with patch("eh30pt.datasets.multinerd.load_dataset", return_value=raw):
        dataset.load()

    assert dataset._sample_examples(0) == []
    assert dataset._sample_examples(1) == [
        "Sentence: John works\nEntity: John\nTag: PER\n"
    ]
    with pytest.raises(ValueError, match="requested more than"):
        dataset._sample_examples(999)


def test_oblique_load_encoder() -> None:
    dataset = Oblique(_stub_tokenizer(), "encoder")
    with patch("eh30pt.datasets.oblique.Dataset", _CsvStub):
        assert dataset.load(n_shot=2) is None

    assert set(dataset.raw_data) == {"train", "validation", "test"}
    assert {row["label"] for row in dataset.raw_data["train"]} <= set(Oblique.label2id)

    row = dataset.data["train"][0]
    assert row["label"] in {0, 1, 2, 3, 4}
    assert "sentence" not in row


def test_oblique_load_decoder_masks_prompt() -> None:
    dataset = Oblique(_stub_tokenizer(), "decoder")
    with patch("eh30pt.datasets.oblique.Dataset", _CsvStub):
        dataset.load(n_shot=2)

    for row in dataset.data["train"].select(range(4)):
        assert len(row["labels"]) == len(row["input_ids"])

    with pytest.raises(ValueError, match="requested more than"):
        dataset._sample_examples(10**6)


def test_collator_with_labels(gpt2_tokenizer: PreTrainedTokenizerFast) -> None:
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [10, 20]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [30]},
    ]

    collator = get_collator(gpt2_tokenizer, "decoder")
    batch = collator(features)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    assert batch["input_ids"].shape[1] % 8 == 0
    assert batch["attention_mask"].shape[1] % 8 == 0
    assert batch["labels"].shape[1] % 8 == 0

    assert batch["input_ids"][0][-1] == gpt2_tokenizer.eos_token_id
    assert batch["attention_mask"][0][-1] == 0
    assert batch["labels"][0][-1] == SENTINEL_TOKEN


def test_collator_seq2seq_pads_labels_to_input_length(
    t5_tokenizer: PreTrainedTokenizerFast,
) -> None:
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [10]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [20, 30]},
    ]

    collator = get_collator(t5_tokenizer, "encoder-decoder")
    batch = collator(features)

    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["labels"][0][-1] == SENTINEL_TOKEN


def test_collator_with_no_labels(gpt2_tokenizer: PreTrainedTokenizerFast) -> None:
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5], "attention_mask": [1, 1]},
    ]

    collator = get_collator(gpt2_tokenizer, "decoder")
    batch = collator(features)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert batch["labels"] is None

    assert batch["input_ids"].shape[1] % 8 == 0
    assert batch["attention_mask"].shape[1] % 8 == 0

    assert batch["input_ids"][0][-1] == gpt2_tokenizer.eos_token_id
    assert batch["attention_mask"][0][-1] == 0


def test_get_dataset_returns_pipeline_without_loading() -> None:
    dataset = get_dataset("multinerd", _tokenizer(), "decoder")

    assert isinstance(dataset, Multinerd)
    assert isinstance(get_dataset("oblique", _tokenizer(), "decoder"), Oblique)


def test_get_dataset_rejects_unimplemented_names() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        get_dataset(cast("DatasetName", "boolq"), _tokenizer(), "decoder")


def test_get_sys_prompt_matches_pipeline() -> None:
    tokenizer = _tokenizer()

    assert get_sys_prompt("oblique", tokenizer, "decoder") == (
        Oblique(tokenizer, "decoder").system_prompt
    )


def test_load_data_returns_loaded_pipeline() -> None:
    frame = DatasetDict({"train": Dataset.from_dict(_multinerd_frame())})
    with patch("eh30pt.datasets.multinerd.load_dataset", return_value=frame):
        dataset = load_data(_stub_tokenizer(), "multinerd", "encoder", 0)

    assert isinstance(dataset, Multinerd)
    assert [row["labels"] for row in dataset.data["train"]] == [0, 1]


def test_load_data_subsets_splits() -> None:
    frame = DatasetDict(
        {
            "train": Dataset.from_dict(_multinerd_frame()),
            "validation": Dataset.from_dict(_multinerd_frame()),
        }
    )
    with patch(
        "eh30pt.datasets.multinerd.load_dataset",
        return_value=frame,
    ):
        dataset = load_data(
            _stub_tokenizer(),
            "multinerd",
            "encoder",
            0,
            n_train_samples=1,
            n_val_samples=1,
        )

    assert len(dataset.data["train"]) == 1
    assert len(dataset.data["validation"]) == 1
