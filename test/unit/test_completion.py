"""Tests for chat completion requests."""

from typing import Any

import pytest
from openai.types.chat import ChatCompletion

from eh30pt.cli.scripts.completion import (
    build_conversation,
    build_response_format,
    extract_content,
    extract_label,
)

_labels: tuple[str, ...] = ("PER", "ORG", "LOC")


def _completion(content: str | None) -> ChatCompletion:
    message: dict[str, Any] = {"role": "assistant"}
    if content is not None:
        message["content"] = content

    return ChatCompletion.model_validate(
        {
            "id": "test",
            "choices": [
                {"finish_reason": "stop", "index": 0, "message": message},
            ],
            "created": 0,
            "model": "test",
            "object": "chat.completion",
        },
    )


def test_build_messages_without_system() -> None:
    assert build_conversation("hi", None) == [{"role": "user", "content": "hi"}]


def test_build_messages_with_system() -> None:
    assert build_conversation("hi", "be brief") == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


def test_extract_content_first_choice() -> None:
    assert extract_content(_completion("Tere")) == "Tere"


def test_extract_content_no_choices() -> None:
    payload = _completion("Tere")
    payload.choices.clear()

    with pytest.raises(ValueError, match="no choices"):
        extract_content(payload)


def test_extract_content_no_content() -> None:
    with pytest.raises(ValueError, match="no content"):
        extract_content(_completion(None))


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


def test_extract_label_valid() -> None:
    assert extract_label(_completion('{"label": "ORG"}'), _labels) == "ORG"


def test_extract_label_invalid_json() -> None:
    with pytest.raises(ValueError, match="invalid json"):
        extract_label(_completion("ORG"), _labels)


def test_extract_label_missing_label() -> None:
    with pytest.raises(ValueError, match="no label"):
        extract_label(_completion('{"tag": "ORG"}'), _labels)


def test_extract_label_unknown_label() -> None:
    with pytest.raises(ValueError, match="unknown label"):
        extract_label(_completion('{"label": "FOO"}'), _labels)
