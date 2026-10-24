"""Tests for folding Label Studio annotations into reason classifier labels."""

from typing import TYPE_CHECKING, Final, cast

import polars as pl

from eh30pt.datasets.discont.classify import DiscontClassifier
from eh30pt.labeling.export.extract import ExtractionFrameSchema, stratify

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from eh30pt.datasets.discont.classify import DiscontClassifierRow

_labels: Final = 9


def test_total_counts_the_split_rows() -> None:
    rows: list[DiscontClassifierRow] = [
        {"id": 1, "text": "t", "phrase": "p", "drug": "d", "label": "Muu põhjus"},
        {"id": 2, "text": "t", "phrase": "p", "drug": "d", "label": "Muu põhjus"},
    ]
    tokenizer = cast("PreTrainedTokenizerFast", None)

    assert DiscontClassifier(rows, tokenizer, "decoder").total() == len(rows)


def test_labels_cover_reasons_and_phrase_flags() -> None:
    assert len(DiscontClassifier.id2label) == _labels
    assert set(DiscontClassifier.label2id) == set(DiscontClassifier.id2label.values())
    assert set(DiscontClassifier.id2label.values()) == set(DiscontClassifier.label2id)


def test_extraction_frame_schema_matches_classifier_labels() -> None:
    """The export schema accepts the English labels mapped to Estonian outputs."""
    [check] = ExtractionFrameSchema.to_schema().columns["label"].checks
    meta = {"Incomplete phrase", "Incorrect phrase"}

    assert set(check.statistics["allowed_values"]) - meta == set(
        DiscontClassifier.en2et
    )
    assert set(DiscontClassifier.id2label.values()) == set(
        DiscontClassifier.en2et.values()
    )


def test_few_shots_cover_every_label() -> None:
    assert {shot.label for shot in DiscontClassifier.few_shots} == set(
        DiscontClassifier.id2label.values()
    )


def test_stratify_streams_splits_interleaved() -> None:
    """Splits must not stream label-blocked, trainers iterate in split order."""
    first, second = "Adverse reactions", "Other"
    n_per_label = 20
    frame = ExtractionFrameSchema.validate(
        pl.DataFrame(
            {
                "project_id": [5] * (2 * n_per_label),
                "task_id": list(range(1, 2 * n_per_label + 1)),
                "annotation_id": list(range(1, 2 * n_per_label + 1)),
                "result_id": [f"res-{i}" for i in range(1, 2 * n_per_label + 1)],
                "annotator": [42] * (2 * n_per_label),
                "from_name": "choices",
                "text": [f"tekst {i}" for i in range(2 * n_per_label)],
                "phrase": [f"fraas {i}" for i in range(2 * n_per_label)],
                "drug": ["enalapriil"] * (2 * n_per_label),
                "reason": ["x"] * (2 * n_per_label),
                "label": [first] * n_per_label + [second] * n_per_label,
            }
        )
    )

    train, eval_, test = stratify(frame, on="label", ratios=(0.8, 0.1, 0.1), seed=0)
    train_labels = train["label"].to_list()

    _expected_train = 2 * round(n_per_label * 0.8)
    assert len(train_labels) == _expected_train
    assert set(train_labels[:8]) == {first, second}

    repeat = stratify(frame, on="label", ratios=(0.8, 0.1, 0.1), seed=0)[0][
        "label"
    ].to_list()
    assert train_labels == repeat

    for split in (eval_, test):
        assert split.height > 0
