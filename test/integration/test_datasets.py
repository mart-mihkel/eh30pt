"""Dataset loading and streaming integration tests."""

from itertools import islice
from typing import TYPE_CHECKING, cast

import pytest

from eh30pt.constants import SENTINEL_TOKEN
from eh30pt.datasets.discontinuation import (
    load_discont_classifier,
    load_discont_extract,
)
from eh30pt.datasets.mixin import DatasetMixin
from eh30pt.modeling.arch import infer_architecture

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from eh30pt.datasets.discontinuation import (
        DeciderRow,
        DiscontClassifierRow,
        DiscontExtractRow,
    )
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
        pytest.skip("tokenizer has no length limit")

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
    data = load_discont_extract(tokenizer, "decoder")
    samples = list(islice(data.test.samples(), 5))

    assert samples
    for sample in samples:
        assert isinstance(sample.user, str)
        assert isinstance(sample.id, int)
        assert sample.label == ""


def test_discont_classifier_streams_sample_source(database: None) -> None:
    """Classifier rows stream from the random sample table when asked."""
    _ = database
    tokenizer = cast("PreTrainedTokenizerFast", None)
    data = load_discont_classifier(tokenizer, "decoder", source="observed")
    samples = list(islice(data.test.samples(), 5))

    assert samples
    assert all(isinstance(sample.id, int) for sample in samples)


def test_discont_splits_iterate_tokenized_rows(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Classifier, decider, and extract splits tokenize in-memory rows."""
    from eh30pt.datasets.discontinuation import (
        DiscontClassifier,
        DiscontDecider,
        DiscontExtract,
    )

    classifier = DiscontClassifier(
        lambda: iter([{"id": 1, "text": "tekkis iiveldus"}]),
        gpt2_tokenizer,
        "decoder",
    )
    decider = DiscontDecider(
        lambda: iter([{"id": 2, "text": "katkestas"}]), gpt2_tokenizer, "decoder"
    )
    extract = DiscontExtract(
        lambda: iter([{"id": 3, "text": "jäeti ravim ära"}]),
        gpt2_tokenizer,
        "decoder",
    )

    for split in (classifier, decider, extract):
        rows = list(iter(split))

        assert rows
        assert set(rows[0]) >= {"input_ids", "labels"}

    assert [sample.id for sample in classifier.samples()] == [1]
    assert [sample.id for sample in decider.samples()] == [2]
    assert [sample.id for sample in extract.samples()] == [3]
    assert extract.response_format()["json_schema"]["name"] == "extraction"

    cls_row: DiscontClassifierRow = {"id": 1, "text": "tekkis iiveldus"}
    dec_row: DeciderRow = {"id": 1, "text": "tekkis iiveldus"}
    ext_row: DiscontExtractRow = {"id": 1, "text": "tekkis iiveldus"}
    assert classifier.tokenize(cls_row, gpt2_tokenizer)["input_ids"]
    assert decider.tokenize(dec_row, gpt2_tokenizer)["input_ids"]
    assert extract.tokenize(ext_row, gpt2_tokenizer)["input_ids"]


def test_join_spans_handles_plain_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-prefixed tags fall through to the plain-label branch."""
    from eh30pt.datasets.multinerd import Multinerd

    bio = dict(Multinerd.id2label_bio)
    bio[99] = "PER"
    monkeypatch.setattr(Multinerd, "id2label_bio", bio)

    tokens, tag_ids = Multinerd._join_spans(["Tallinn"], [99])

    assert tokens == ["Tallinn"]
    assert tag_ids == [Multinerd.label2id["PER"]]
