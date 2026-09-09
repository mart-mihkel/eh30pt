"""Tests for chat completion requests."""

import pytest

from eh30pt.cli.scripts.completion import (
    build_messages,
    build_response_format,
    classification_prompt,
    extract_content,
    extract_label,
)

_labels: tuple[str, ...] = ("PER", "ORG", "LOC")


def test_build_messages_without_system() -> None:
    assert build_messages("hi", None) == [{"role": "user", "content": "hi"}]


def test_build_messages_with_system() -> None:
    assert build_messages("hi", "be brief") == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


def test_extract_content_first_choice() -> None:
    content = extract_content(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Tere",
                    }
                }
            ]
        }
    )

    assert content == "Tere"


def test_extract_content_no_choices() -> None:
    with pytest.raises(ValueError, match="no choices"):
        extract_content({"choices": []})


def test_build_response_format_label_enum() -> None:
    assert build_response_format(_labels) == {
        "type": "json_schema",
        "json_schema": {
            "name": "label",
            "schema": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": ["PER", "ORG", "LOC"],
                    },
                },
                "required": ["label"],
            },
        },
    }


def test_classification_prompt_lists_labels() -> None:
    prompt = classification_prompt(_labels)

    assert '{"label": "<TAG>"}' in prompt
    assert "PER, ORG, LOC" in prompt


def test_extract_label_valid() -> None:
    label = extract_label(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"label": "ORG"}',
                    }
                }
            ]
        },
        _labels,
    )

    assert label == "ORG"


def test_extract_label_invalid_json() -> None:
    with pytest.raises(ValueError, match="invalid json"):
        extract_label(
            {"choices": [{"message": {"role": "assistant", "content": "ORG"}}]},
            _labels,
        )


def test_extract_label_missing_label() -> None:
    with pytest.raises(ValueError, match="no label"):
        extract_label(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"tag": "ORG"}',
                        }
                    }
                ]
            },
            _labels,
        )


def test_extract_label_unknown_label() -> None:
    with pytest.raises(ValueError, match="unknown label"):
        extract_label(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"label": "FOO"}',
                        }
                    }
                ]
            },
            _labels,
        )
