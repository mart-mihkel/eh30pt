"""Tests for chat conversation building."""

from eh30pt.datasets.registry import get_few_shots, get_sys_prompt
from eh30pt.openai.complete import CompletionClient


def test_build_messages_without_system() -> None:
    user = get_few_shots("multinerd", 1)[0].user

    assert CompletionClient.build_conversation(user, None) == [
        {"role": "user", "content": user}
    ]


def test_build_messages_with_system() -> None:
    system = get_sys_prompt("multinerd")
    user = get_few_shots("multinerd", 1)[0].user

    assert CompletionClient.build_conversation(user, system) == [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def test_build_messages_with_examples() -> None:
    examples = get_few_shots("oblique", 2)

    assert CompletionClient.build_conversation(examples=examples) == [
        {"role": "user", "content": examples[0].user},
        {"role": "assistant", "content": examples[0].label},
        {"role": "user", "content": examples[1].user},
        {"role": "assistant", "content": examples[1].label},
    ]


def test_build_messages_with_system_examples_and_prompt() -> None:
    system = get_sys_prompt("oblique")
    examples = get_few_shots("oblique", 1)
    user = get_few_shots("oblique", 2)[1].user

    assert CompletionClient.build_conversation(
        user,
        system=system,
        examples=examples,
    ) == [
        {"role": "system", "content": system},
        {"role": "user", "content": examples[0].user},
        {"role": "assistant", "content": examples[0].label},
        {"role": "user", "content": user},
    ]
