"""Llama server integration tests with cassette-recorded responses."""

from typing import Final

import pytest

from eh30pt.cli.scripts.completion import main
from eh30pt.datasets.registry import get_labels

_base_url: Final = "http://localhost:8080"
_match_on: Final = ["method", "scheme", "host", "port", "path", "body"]


@pytest.mark.vcr(match_on=_match_on)
def test_free_text() -> None:
    content = main(
        "What is 2+2? Answer in one word.",
        base_url=_base_url,
        system="You are terse.",
        max_tokens=300,
    )

    assert content.strip().rstrip(".").lower() in {"4", "four"}


@pytest.mark.vcr(match_on=_match_on)
def test_structured_label() -> None:
    content = main(
        "Sentence: Barack Obama was president.\nEntity: Barack Obama\nTag:",
        base_url=_base_url,
        system="You are a NER tagger.",
        max_tokens=1024,
        labels=get_labels("multinerd"),
    )

    assert content == "PER"
