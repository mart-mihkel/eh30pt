"""Tests for prompting, few-shot formatting, and dataset registries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest

from eh30pt.datasets.discont.classify import DiscontClassifier
from eh30pt.datasets.discont.extract import (
    DiscontExtract,
    DiscontExtractRow,
    _no_rows,
)
from eh30pt.datasets.multinerd import Multinerd
from eh30pt.datasets.oblique import Obl
from eh30pt.datasets.registry import (
    assert_classification,
    assert_extraction,
    get_extract_fields,
    get_extract_shots,
    get_few_shots,
    get_labels,
    get_positive_label,
    get_response_schema,
    get_sys_prompt,
    get_task,
    load_data,
)

if TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from transformers import PreTrainedTokenizerFast

    from eh30pt.types import Architecture, DatasetName


def test_oblique_format_user_prompt() -> None:
    assert (
        Obl._format_user_prompt("Tere maailm", "Tere")
        == "Lause: Tere maailm\nFraas: Tere\nKategooria:"
    )


def test_multinerd_format_user_prompt() -> None:
    assert (
        Multinerd._format_user_prompt("Cats run fast", "Cats")
        == "Sentence: Cats run fast\nEntity: Cats\nTag:"
    )


def test_few_shots_cover_labels() -> None:
    assert {shot.label for shot in Obl.few_shots} == set(Obl.id2label.values())
    assert len(Obl.few_shots) == 2 * len(Obl.id2label)

    assert {shot.label for shot in Multinerd.few_shots} == set(
        Multinerd.id2label.values()
    )

    assert {shot.label for shot in DiscontClassifier.few_shots} == set(
        DiscontClassifier.id2label.values()
    )


def test_get_few_shots() -> None:
    shots = get_few_shots("oblique", 5)
    assert [shot.label for shot in shots] == [
        "seotud",
        "vaba",
        "ebaloomulik",
        "liigne koma",
        "kaheldav",
    ]

    with pytest.raises(ValueError, match="only 10 defined"):
        get_few_shots("oblique", 11)


def test_discont_extract_few_shots_are_extraction_objects() -> None:
    _defined = 5
    shots = get_few_shots("discont-extract/trigger", _defined)

    assert len(shots) == _defined
    for shot in shots:
        assert set(json.loads(shot.label)) == set(
            get_extract_fields("discont-extract/trigger")
        )

    with pytest.raises(ValueError, match="only 5 defined"):
        get_few_shots("discont-extract/trigger", _defined + 1)


def test_few_shots_derive_from_extract_shots() -> None:
    pairs = list(
        zip(
            DiscontExtract.few_shots,
            DiscontExtract.extract_shots,
            strict=True,
        )
    )

    assert len(pairs) == len(DiscontExtract.extract_shots)
    for few, extract in pairs:
        assert few.user == extract.user
        assert json.loads(few.label) == extract.extractions


def test_registry() -> None:
    _obl_fewshots = 5
    _multinerd_labels = 15
    _classifier_labels = 9

    assert get_labels("multinerd")[:3] == ("PER", "ORG", "LOC")
    assert len(get_labels("multinerd")) == _multinerd_labels
    assert len(get_labels("discont-classifier/extract-labels")) == _classifier_labels

    assert get_sys_prompt("oblique") == Obl.system_prompt
    assert get_sys_prompt("multinerd") == Multinerd.system_prompt
    assert (
        get_sys_prompt("discont-classifier/extract-labels")
        == DiscontClassifier.system_prompt
    )
    assert get_sys_prompt("discont-extract/trigger") == DiscontExtract.system_prompt

    assert len(get_few_shots("oblique", 5)) == _obl_fewshots

    with pytest.raises(ValueError, match="only 10 defined"):
        get_few_shots("oblique", 11)


def test_get_positive_label_only_for_binary_datasets() -> None:
    assert get_positive_label("discont-binary/extract-labels", "decoder") == "jah"
    assert get_positive_label("discont-binary/extract-labels", "encoder") == 0
    assert (
        get_positive_label("discont-binary/extract-labels", "encoder-decoder") == "jah"
    )

    assert get_positive_label("oblique", "decoder") is None
    assert get_positive_label("multinerd", "encoder") is None
    assert get_positive_label("discont-classifier/extract-labels", "decoder") is None
    assert get_positive_label("discont-extract/trigger", "decoder") is None

    with pytest.raises(ValueError, match="invalid dataset"):
        get_positive_label(cast("DatasetName", "unknown"), "decoder")


@pytest.mark.parametrize(
    "dataset",
    [
        "multinerd",
        "oblique",
        "discont-classifier/extract-labels",
        "discont-binary/extract-labels",
        "discont-extract/trigger",
    ],
)
def test_get_sys_prompt_is_short_for_encoder(dataset: DatasetName) -> None:
    encoder_prompt = get_sys_prompt(dataset, "encoder")

    assert encoder_prompt
    assert len(encoder_prompt) < len(get_sys_prompt(dataset))
    assert get_sys_prompt(dataset, "decoder") == get_sys_prompt(dataset)
    assert get_sys_prompt(dataset, "encoder-decoder") == get_sys_prompt(dataset)


def test_get_response_schema() -> None:
    schema = get_response_schema("multinerd")

    assert schema["type"] == "json_schema"
    assert schema["json_schema"]["name"] == "label"
    assert schema["json_schema"]["schema"] == {
        "type": "string",
        "enum": list(Multinerd.id2label.values()),
    }
    assert get_response_schema("oblique")["json_schema"]["schema"] == {
        "type": "string",
        "enum": list(Obl.id2label.values()),
    }

    extract = get_response_schema("discont-extract/trigger")

    assert extract["json_schema"]["name"] == "extraction"
    assert (
        extract["json_schema"]["schema"]
        == DiscontExtract.response_format()["json_schema"]["schema"]
    )

    with pytest.raises(ValueError, match="invalid dataset"):
        get_response_schema(cast("DatasetName", "unknown"))


def test_get_task_marks_extraction_dataset() -> None:

    assert get_task("multinerd") == "classification"
    assert get_task("oblique") == "classification"
    assert get_task("discont-classifier/extract-labels") == "classification"
    assert get_task("discont-extract/trigger") == "extraction"

    with pytest.raises(ValueError, match="invalid dataset"):
        get_task(cast("DatasetName", "unknown"))


def test_get_labels_rejects_extraction_dataset() -> None:
    with pytest.raises(ValueError, match="has no labels"):
        get_labels("discont-extract/trigger")


def test_require_classification_rejects_extraction_dataset() -> None:
    assert_classification("oblique", "training needs labels")

    with pytest.raises(ValueError, match="extraction dataset"):
        assert_classification("discont-extract/trigger", "training needs labels")


def test_require_extraction_rejects_classification_dataset() -> None:
    assert_extraction("discont-extract/trigger", "langextract needs extraction fields")

    with pytest.raises(ValueError, match="classification dataset"):
        assert_extraction("oblique", "langextract needs extraction fields")


def test_get_extract_shots() -> None:
    _defined = 5
    shots = get_extract_shots("discont-extract/trigger", _defined)

    assert len(shots) == _defined
    assert all(
        set(shot.extractions) == set(get_extract_fields("discont-extract/trigger"))
        for shot in shots
    )

    with pytest.raises(ValueError, match="only 5 defined"):
        get_extract_shots("discont-extract/trigger", _defined + 1)

    with pytest.raises(ValueError, match="has no extraction examples"):
        get_extract_shots("oblique", 1)


def test_get_extract_fields() -> None:
    assert get_extract_fields("discont-extract/trigger") == ("phrase", "reason", "drug")

    with pytest.raises(ValueError, match="has no extraction fields"):
        get_extract_fields("oblique")


def test_response_schemas_serialize_to_json() -> None:
    datasets: list[DatasetName] = [
        "multinerd",
        "oblique",
        "discont-classifier/extract-labels",
        "discont-extract/trigger",
    ]

    for dataset in datasets:
        assert json.loads(json.dumps(get_response_schema(dataset))) == (
            get_response_schema(dataset)
        )


@pytest.mark.parametrize("dataset", ["multinerd", "oblique"])
def test_load_data_rejects_a_data_path_for_static_datasets(
    dataset: DatasetName,
) -> None:
    tokenizer = cast("PreTrainedTokenizerFast", None)
    with pytest.raises(ValueError, match="labels file is only read by"):
        load_data(tokenizer, dataset, "decoder", data_path=Path("labels.parquet"))


def test_load_data_rejects_a_data_path_for_extract_datasets() -> None:
    tokenizer = cast("PreTrainedTokenizerFast", None)

    with pytest.raises(ValueError, match="labels file is only read by"):
        load_data(
            tokenizer,
            "discont-extract/trigger",
            "decoder",
            data_path=Path("labels.parquet"),
        )


def test_load_data_requires_a_data_path_for_the_classifier() -> None:
    tokenizer = cast("PreTrainedTokenizerFast", None)

    with pytest.raises(ValueError, match="needs a labels file"):
        load_data(tokenizer, "discont-classifier/extract-labels", "decoder")


def test_load_data_streams_the_test_split_only() -> None:
    tokenizer = cast("PreTrainedTokenizerFast", None)
    data = load_data(tokenizer, "discont-extract/trigger", "decoder")

    assert list(data.train) == []
    assert list(data.eval) == []


def test_discont_samples_carry_row_id() -> None:

    rows: list[DiscontExtractRow] = [
        {"id": 42, "text": "katkestas"},
        {"id": 7, "text": "lõpetas"},
    ]
    tokenizer = cast("PreTrainedTokenizerFast", None)
    extract = DiscontExtract(lambda: iter(rows), tokenizer, "decoder")

    assert [(sample.id, sample.user) for sample in extract.samples()] == [
        (42, "katkestas"),
        (7, "lõpetas"),
    ]


@pytest.mark.parametrize("arch", ["encoder", "encoder-decoder"])
def test_load_data_rejects_non_decoder_arch_for_extract(arch: Architecture) -> None:
    tokenizer = cast("PreTrainedTokenizerFast", None)

    with pytest.raises(ValueError, match="not supported"):
        load_data(tokenizer, "discont-extract/trigger", arch)


def test_get_labels_rejects_invalid_dataset() -> None:
    with pytest.raises(ValueError, match="invalid dataset"):
        get_labels(cast("DatasetName", "unknown"))


def test_get_sys_prompt_rejects_invalid_dataset() -> None:
    with pytest.raises(ValueError, match="invalid dataset"):
        get_sys_prompt(cast("DatasetName", "unknown"))


def test_get_few_shots_rejects_invalid_dataset() -> None:
    with pytest.raises(ValueError, match="invalid dataset"):
        get_few_shots(cast("DatasetName", "unknown"), 1)


def test_get_few_shots_covers_datasets() -> None:
    _n_shots: Final = 2

    assert len(get_few_shots("discont-classifier/extract-labels", _n_shots)) == _n_shots
    assert len(get_few_shots("multinerd", _n_shots)) == _n_shots

    with pytest.raises(ValueError, match="only 9 defined"):
        get_few_shots(
            "discont-classifier/extract-labels",
            len(DiscontClassifier.few_shots) + 1,
        )


def test_dataset_sources_cover_discont_tables() -> None:
    from mlflow.data.code_dataset_source import CodeDatasetSource

    from eh30pt.tracking.datasets import dataset_input, dataset_source

    assert isinstance(
        dataset_source("discont-classifier/extract-labels"), CodeDatasetSource
    )
    assert isinstance(dataset_source("discont-extract/trigger"), CodeDatasetSource)
    assert dataset_input("oblique").profile is None


def test_load_data_rejects_invalid_dataset() -> None:
    tokenizer = cast("PreTrainedTokenizerFast", None)

    with pytest.raises(ValueError, match="invalid dataset"):
        load_data(tokenizer, cast("DatasetName", "unknown"), "decoder")


def test_dataset_input_rejects_invalid_dataset() -> None:
    from eh30pt.tracking.datasets import dataset_input, dataset_source

    with pytest.raises(ValueError, match="invalid dataset"):
        dataset_source(cast("DatasetName", "unknown"))

    with pytest.raises(ValueError, match="invalid dataset"):
        dataset_input(cast("DatasetName", "unknown"))


def test_dataset_input_covers_extract_dataset() -> None:
    from eh30pt.tracking.datasets import dataset_input

    assert dataset_input("discont-extract/trigger").name == "discont-extract/trigger"


def test_mixin_base_methods_raise() -> None:
    from eh30pt.datasets.mixin import DatasetMixin

    mixin = DatasetMixin[None]()

    with pytest.raises(NotImplementedError):
        mixin.tokenize(None, cast("PreTrainedTokenizerFast", None))

    with pytest.raises(NotImplementedError):
        list(mixin.samples())

    with pytest.raises(TypeError, match="iterate instead"):
        mixin["train"]

    assert mixin.set_epoch(3) is None


def test_discont_tokenize_rejects_too_many_shots() -> None:

    tokenizer = cast("PreTrainedTokenizerFast", None)

    with pytest.raises(ValueError, match="more few shot examples"):
        DiscontClassifier([], tokenizer, "decoder").tokenize(
            {"id": 1, "text": "x", "phrase": "x", "drug": "y", "label": "Muu põhjus"},
            tokenizer,
            n_shot=100,
        )

    with pytest.raises(ValueError, match="more few shot examples"):
        DiscontExtract(lambda: iter([]), tokenizer, "decoder").tokenize(
            {"id": 1, "text": "x"}, tokenizer, n_shot=100
        )


def test_multinerd_tokenize_rejects_too_many_shots() -> None:
    from eh30pt.datasets.multinerd import Multinerd

    tokenizer = cast("PreTrainedTokenizerFast", None)
    split = Multinerd(cast("Dataset", None), tokenizer, "decoder")

    with pytest.raises(ValueError, match="more few shot examples"):
        split.tokenize(
            {"sentence": "Hi", "entity": "Hi", "label": "PER"},
            tokenizer,
            n_shot=1000,
        )


def test_no_extract_row_factory_is_empty() -> None:
    assert list(_no_rows()) == []
