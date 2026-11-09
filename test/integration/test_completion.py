"""Completion integration tests against llama, backed by recorded cassettes."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from eh30pt.datasets.registry import get_few_shots, get_sys_prompt
from eh30pt.openai.complete import OpenaiClient
from eh30pt.settings import ServerSettings

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam


def _conversation() -> list[ChatCompletionMessageParam]:
    """Build a real multinerd conversation from registry prompts and shots."""
    shots = get_few_shots("multinerd", 2)
    return OpenaiClient.build_conversation(
        shots[1].user,
        system=get_sys_prompt("multinerd"),
        examples=shots[:1],
    )


async def _real_completion() -> ChatCompletion:
    """Fetch a real completion from the served model."""
    client = OpenaiClient(ServerSettings(provider="llama"), seed=0)
    model = (await client.model()).unwrap()
    return await client._client.chat.completions.create(
        model=model,
        messages=_conversation(),
    )


@pytest.mark.vcr
def test_complete_returns_content() -> None:
    client = OpenaiClient(ServerSettings(provider="llama"), seed=0)
    content = asyncio.run(client.complete(_conversation())).unwrap()

    assert isinstance(content, str)
    assert content.strip()


@pytest.mark.vcr
def test_extract_content_reads_first_choice() -> None:
    completion = asyncio.run(_real_completion())

    assert completion.choices
    assert (
        OpenaiClient._extract_content(completion)
        == completion.choices[0].message.content
    )


@pytest.mark.vcr
def test_extract_content_rejects_empty_choices() -> None:
    completion = asyncio.run(_real_completion())
    completion.choices.clear()

    with pytest.raises(ValueError, match="no choices"):
        OpenaiClient._extract_content(completion)


@pytest.mark.vcr
def test_extract_content_rejects_missing_content() -> None:
    completion = asyncio.run(_real_completion())
    completion.choices[0].message.content = None

    with pytest.raises(ValueError, match="no content"):
        OpenaiClient._extract_content(completion)
