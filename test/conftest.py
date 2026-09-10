"""Test configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping

models: Final[Mapping[str, str]] = {
    "bert": "hf-internal-testing/tiny-random-bert",
    "modernbert": "hf-internal-testing/tiny-random-ModernBertForSequenceClassification",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "gpt-neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "qwen": "tiny-random/qwen3",
    "gemma": "tiny-random/gemma-3",
    "t5": "hf-internal-testing/tiny-random-t5",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        default=False,
        action="store_true",
        help="run tests marked as slow",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="pass --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
