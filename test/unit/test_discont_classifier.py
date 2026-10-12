"""Tests for folding Label Studio annotations into reason classifier labels."""

from typing import Final

from eh30pt.datasets.discontinuation.classify import DiscontClassifier, _map_label

_reasons: Final = 8


def test_map_label_accepts_every_classifier_label() -> None:
    for label in DiscontClassifier.id2label.values():
        assert _map_label(label) == label


def test_map_label_rejects_anything_else() -> None:
    assert _map_label("Something new") is None
    assert _map_label("") is None


def test_labels_are_the_eight_reasons() -> None:
    assert len(DiscontClassifier.id2label) == _reasons
    assert set(DiscontClassifier.label2id) == set(DiscontClassifier.id2label.values())
    assert set(DiscontClassifier.id2label.values()) == set(DiscontClassifier.label2id)


def test_few_shots_cover_every_label() -> None:
    assert {shot.label for shot in DiscontClassifier.few_shots} == set(
        DiscontClassifier.id2label.values()
    )
