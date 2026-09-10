"""Tests for chat completion requests."""

from typing import TYPE_CHECKING

import pytest
from openai.types.chat import ChatCompletion

from eh30pt.openai.complete import CompletionClient
from eh30pt.types import FewShot

if TYPE_CHECKING:
    from collections.abc import Callable

_labels: tuple[str, ...] = ("PER", "ORG", "LOC")


@pytest.fixture
def make_completion() -> Callable[[str | None], ChatCompletion]:
    """Return a factory building chat completions with the given content."""

    def _factory(content: str | None) -> ChatCompletion:
        message: dict[str, str] = {"role": "assistant"}
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

    return _factory


def test_build_messages_without_system() -> None:
    assert CompletionClient.build_conversation("hi", None) == [
        {"role": "user", "content": "hi"}
    ]


def test_build_messages_with_system() -> None:
    assert CompletionClient.build_conversation("hi", "be brief") == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


def test_build_messages_with_examples() -> None:
    examples = [
        FewShot(user="a", label="x"),
        FewShot(user="b", label="y"),
    ]

    assert CompletionClient.build_conversation(examples=examples) == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "y"},
    ]


def test_build_messages_with_system_examples_and_prompt() -> None:
    examples = [FewShot(user="a", label="x")]

    assert CompletionClient.build_conversation(
        "hi",
        system="be brief",
        examples=examples,
    ) == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": "hi"},
    ]


def test_extract_content_first_choice(
    make_completion: Callable[[str | None], ChatCompletion],
) -> None:
    assert CompletionClient._extract_content(make_completion("Tere")) == "Tere"


def test_extract_content_no_choices(
    make_completion: Callable[[str | None], ChatCompletion],
) -> None:
    payload = make_completion("Tere")
    payload.choices.clear()

    with pytest.raises(ValueError, match="no choices"):
        CompletionClient._extract_content(payload)


def test_extract_content_no_content(
    make_completion: Callable[[str | None], ChatCompletion],
) -> None:
    with pytest.raises(ValueError, match="no content"):
        CompletionClient._extract_content(make_completion(None))
