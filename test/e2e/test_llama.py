"""End to end eh workflows with cassette-recorded llama responses."""

from typing import TYPE_CHECKING, Final

import pytest
from typer.testing import CliRunner

from eh30pt.cli.llama import app

if TYPE_CHECKING:
    from typer.testing import Result

_match_on: Final = ["method", "scheme", "host", "port", "path", "body"]


def _invoke(args: list[str]) -> Result:
    return CliRunner().invoke(app, args)


@pytest.mark.vcr(match_on=_match_on)
def test_completion_prints_completion() -> None:
    result = _invoke(
        [
            "completion",
            "What is 2+2? Answer in one word.",
            "--system",
            "You are terse.",
            "--max-tokens",
            "300",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "4" in result.output or "Four" in result.output


@pytest.mark.vcr(match_on=_match_on)
def test_completion_dataset_prints_label() -> None:
    result = _invoke(
        [
            "completion",
            "Sentence: Barack Obama was president.\nEntity: Barack Obama\nTag:",
            "--system",
            "You are a NER tagger.",
            "--dataset",
            "multinerd",
            "--max-tokens",
            "1024",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "PER" in result.output.split()
