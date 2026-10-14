"""Tests for folding Label Studio annotations into reason classifier labels."""

from typing import Final

from eh30pt.datasets.discontinuation.classify import DiscontClassifier
from eh30pt.labeling.loaders.extract import ExtractionFrameSchema

_labels: Final = 8


def test_labels_cover_reasons_and_phrase_flags() -> None:
    assert len(DiscontClassifier.id2label) == _labels
    assert set(DiscontClassifier.label2id) == set(DiscontClassifier.id2label.values())
    assert set(DiscontClassifier.id2label.values()) == set(DiscontClassifier.label2id)


def test_extraction_frame_schema_matches_classifier_labels() -> None:
    """The export schema accepts exactly the classifier labels, no drift."""
    [check] = ExtractionFrameSchema.to_schema().columns["label"].checks

    assert set(check.statistics["allowed_values"]) == set(
        DiscontClassifier.id2label.values()
    )


def test_few_shots_cover_every_label() -> None:
    assert {shot.label for shot in DiscontClassifier.few_shots} == set(
        DiscontClassifier.id2label.values()
    )
