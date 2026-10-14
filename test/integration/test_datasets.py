"""Dataset loading and streaming integration tests."""

from itertools import islice
from typing import TYPE_CHECKING, cast

import pytest

from eh30pt.constants import SENTINEL_TOKEN
from eh30pt.datasets.discontinuation.classify import DiscontClassifier
from eh30pt.datasets.discontinuation.extract import (
    DiscontExtract,
    DiscontExtractRow,
    load_discont_extract,
)
from eh30pt.datasets.mixin import DatasetMixin
from eh30pt.datasets.registry import load_data
from eh30pt.modeling.arch import infer_architecture

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from eh30pt.datasets.discontinuation.classify import DiscontClassifierRow
    from test.integration.conftest import ModelSetup


def test_splits_stream_tokenized_rows(model_setup: ModelSetup) -> None:
    expected = {"input_ids", "attention_mask", "labels", "truncated"}
    splits = (
        model_setup.splits.train,
        model_setup.splits.eval,
        model_setup.splits.test,
    )

    for split in splits:
        rows = list(islice(split, 4))

        assert rows
        assert all(expected <= set(row) for row in rows)


def test_labels_match_architecture(model_setup: ModelSetup) -> None:
    arch = infer_architecture(model_setup.config)
    rows = list(islice(model_setup.splits.train, 8))

    assert rows
    if arch == "encoder":
        assert all(isinstance(row["labels"], int) for row in rows)
        assert all(0 <= row["labels"] < len(model_setup.labels) for row in rows)
    else:
        assert all(isinstance(row["labels"], list) for row in rows)
        assert any(label != SENTINEL_TOKEN for row in rows for label in row["labels"])


def test_truncation_flags(model_setup: ModelSetup) -> None:
    max_length = DatasetMixin._model_max_len(model_setup.tokenizer, 0)
    if max_length is None:
        pytest.xfail("tokenizer has no length limit")

    for row in islice(model_setup.splits.train, 8):
        ids = row["input_ids"]
        assert row["truncated"] == (len(ids) >= max_length)


def test_samples_have_valid_labels(model_setup: ModelSetup) -> None:
    split = model_setup.splits.test
    samples = list(islice(split.samples(), 2))

    assert samples
    for sample in samples:
        assert sample.user
        assert sample.label in model_setup.labels


def test_discont_extract_streams_seeded_rows(database: None) -> None:
    """Extract candidates stream from seeded postgres with row ids."""
    _ = database
    tokenizer = cast("PreTrainedTokenizerFast", None)
    data = load_discont_extract(tokenizer, "decoder", source="trigger")
    samples = list(islice(data.test.samples(), 5))

    assert samples
    for sample in samples:
        assert isinstance(sample.user, str)
        assert isinstance(sample.id, int)
        assert sample.label == ""


def test_discont_extract_streams_observed_source(database: None) -> None:
    """Extract candidates stream from the unfiltered event table when asked."""
    _ = database
    tokenizer = cast("PreTrainedTokenizerFast", None)
    data = load_discont_extract(tokenizer, "decoder", source="observed")
    samples = list(islice(data.test.samples(), 5))

    assert samples
    assert all(isinstance(sample.id, int) for sample in samples)


def test_discont_splits_iterate_tokenized_rows(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Classifier and extract splits tokenize rows from registry examples."""
    from eh30pt.datasets.registry import get_labels

    real_shot = DiscontExtract.extract_shots[0]
    real_label = get_labels("discont-classifier/extract-labels")[0]
    classifier = DiscontClassifier(
        [
            {
                "id": 1,
                "text": real_shot.user,
                "phrase": real_shot.extractions["phrase"],
                "drug": real_shot.extractions["drug"],
                "label": real_label,
            }
        ],
        gpt2_tokenizer,
        "decoder",
    )

    extract_rows: list[DiscontExtractRow] = [
        {"id": idx, "text": shot.user}
        for idx, shot in enumerate(DiscontExtract.extract_shots[:1])
    ]
    extract = DiscontExtract(
        lambda: iter(extract_rows),
        gpt2_tokenizer,
        "decoder",
    )

    for split in (classifier, extract):
        rows = list(iter(split))

        assert rows
        assert set(rows[0]) >= {"input_ids", "labels"}

    assert [sample.id for sample in classifier.samples()] == [1]
    assert [sample.id for sample in extract.samples()] == [
        row["id"] for row in extract_rows
    ]
    assert extract.response_format()["json_schema"]["name"] == "extraction"

    cls_row: DiscontClassifierRow = {
        "id": 1,
        "text": real_shot.user,
        "phrase": real_shot.extractions["phrase"],
        "drug": real_shot.extractions["drug"],
        "label": real_label,
    }

    ext_row: DiscontExtractRow = {"id": 1, "text": real_shot.user}
    assert classifier.tokenize(cls_row, gpt2_tokenizer)["input_ids"]
    assert extract.tokenize(ext_row, gpt2_tokenizer)["input_ids"]


def test_join_spans_assembles_real_entities() -> None:
    """BIO spans from a real MultiNERD row join into entity strings."""
    from datasets.load import load_dataset
    from datasets.utils.info_utils import VerificationMode

    from eh30pt.datasets.multinerd import Multinerd

    raw = load_dataset(
        "Babelscape/multinerd",
        split="train[:1]",
        verification_mode=VerificationMode.NO_CHECKS,
    )
    row = raw[0]
    tokens, tag_ids = Multinerd._join_spans(row["tokens"], row["ner_tags"])

    assert len(tokens) == len(tag_ids)
    assert len(tokens) == len(row["tokens"]) or len(tokens) < len(row["tokens"])
    assert all(tag_id == -1 or tag_id in Multinerd.id2label for tag_id in tag_ids)


def test_load_data_omits_source_for_static_datasets(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    data = load_data(gpt2_tokenizer, "oblique", "decoder")

    assert list(data.test.samples())
