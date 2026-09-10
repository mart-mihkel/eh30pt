"""Tests for langextract example building and prediction assembly."""

import json

from langextract.data import AnnotatedDocument, Extraction

from eh30pt.cli.scripts.langextract import (
    assemble_prediction,
    build_examples,
    build_model_config,
)
from eh30pt.types import ExtractShot

_fields: tuple[str, ...] = (
    "drug_stop_phrase",
    "reason_for_stopping",
    "drug_name",
)


def _shot() -> ExtractShot:
    return ExtractShot(
        user="Patsient katkestas Nebileti tarvitamise.",
        extractions={
            "drug_stop_phrase": "Patsient katkestas Nebileti tarvitamise.",
            "reason_for_stopping": "",
            "drug_name": "Nebilet",
        },
    )


def _document(spans: dict[str, str]) -> AnnotatedDocument:
    return AnnotatedDocument(
        text="input",
        extractions=[
            Extraction(extraction_class=field, extraction_text=span)
            for field, span in spans.items()
        ],
    )


def test_build_examples_empty() -> None:
    assert build_examples([]) == []


def test_build_examples_pairs_text_with_field_spans() -> None:
    examples = build_examples([_shot()])

    assert [example.text for example in examples] == [_shot().user]
    assert {
        extraction.extraction_class: extraction.extraction_text
        for extraction in examples[0].extractions
    } == _shot().extractions


def test_assemble_prediction_returns_field_object() -> None:
    prediction = assemble_prediction(_document(dict(_shot().extractions)), _fields)

    assert prediction is not None
    assert json.loads(prediction) == _shot().extractions


def test_assemble_prediction_fills_missing_fields() -> None:
    prediction = assemble_prediction(_document({"drug_name": "Nebilet"}), _fields)

    assert prediction is not None
    assert json.loads(prediction) == {
        "drug_stop_phrase": "",
        "reason_for_stopping": "",
        "drug_name": "Nebilet",
    }


def test_assemble_prediction_drops_unknown_classes() -> None:
    prediction = assemble_prediction(
        _document({"drug_name": "Nebilet", "other": "x"}), _fields
    )

    assert prediction is not None
    assert "other" not in json.loads(prediction)


def test_assemble_prediction_without_extractions_is_invalid() -> None:
    result = AnnotatedDocument(text="nothing", extractions=[])

    assert assemble_prediction(result, _fields) is None


def test_build_model_config_targets_server() -> None:
    config = build_model_config("llama", "http://localhost:8080", "no-auth")

    assert config.model_id == "llama"
    assert config.provider == "openai"
    assert config.provider_kwargs["base_url"] == "http://localhost:8080"
    assert config.provider_kwargs["api_key"] == "no-auth"
