"""Check CLI option choices stay in sync with Literal types."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import click
import pytest

from eh30pt.cli.options.common import (
    arch_option,
    dataset_option,
    inferred_dataset_option,
    log_level_option,
)
from eh30pt.cli.options.server import (
    completion_dataset_option,
    data_source_option,
    provider_option,
)
from eh30pt.cli.options.training import prefix_init_option
from eh30pt.types import (
    Architecture,
    DatasetName,
    DataSource,
    LogLevel,
    PrefixInit,
    Provider,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _literal_values(type_alias: type) -> frozenset[str]:
    """Extract the set of string values from a PEP 695 Literal type alias."""
    args = get_args(type_alias.__value__)  # ty: ignore[unresolved-attribute]
    return frozenset(args)


def _choice_values(option: Callable[..., object]) -> frozenset[str]:
    """Apply an option decorator to a dummy function and extract click.Choice values."""

    def dummy() -> None:
        pass

    option(dummy)
    params = getattr(dummy, "__click_params__", [])
    assert len(params) == 1, f"expected one param, got {len(params)}"
    param_type = params[0].type
    assert isinstance(param_type, click.Choice), "option does not use click.Choice"
    return frozenset(param_type.choices)


_PAIRS = [
    ("dataset_option", dataset_option, DatasetName),
    ("inferred_dataset_option", inferred_dataset_option, DatasetName),
    ("completion_dataset_option", completion_dataset_option, DatasetName),
    ("log_level_option", log_level_option, LogLevel),
    ("arch_option", arch_option, Architecture),
    ("data_source_option", data_source_option, DataSource),
    ("provider_option", provider_option, Provider),
    ("prefix_init_option", prefix_init_option, PrefixInit),
]


@pytest.mark.parametrize(
    ("option", "type_alias"),
    [(opt, alias) for _, opt, alias in _PAIRS],
    ids=[name for name, _, _ in _PAIRS],
)
def test_choices_match_literal(option: Callable[..., object], type_alias: type) -> None:
    """Option click.Choice choices must match their Literal type values."""
    expected = _literal_values(type_alias)
    actual = _choice_values(option)
    assert actual == expected, (
        f"choices {sorted(actual)} do not match literal values {sorted(expected)}"
    )
