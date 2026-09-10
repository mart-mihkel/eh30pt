"""Shared fixtures for integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import datasets
import pytest

from test.conftest import models

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from datasets.dataset_dict import DatasetDict
    from datasets.splits import Split
    from transformers import PreTrainedConfig, PreTrainedModel, PreTrainedTokenizerFast

    from eh30pt.types import DatasetName, DataSplits

datasets.disable_caching()

_datasets: tuple[DatasetName, ...] = ("multinerd", "oblique")


class ModelSetup(NamedTuple):
    """One model under test with its config, tokenizer, labels, splits, and weights."""

    config: PreTrainedConfig
    tokenizer: PreTrainedTokenizerFast
    labels: tuple[str, ...]
    splits: DataSplits
    model: PreTrainedModel


def _make_config_fixture(
    path: str,
) -> Callable[[], PreTrainedTokenizerFast]:
    @pytest.fixture(scope="session")
    def fixture() -> PreTrainedTokenizerFast:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(path)

    return fixture


def _make_tokenizer_fixture(
    path: str,
) -> Callable[[], PreTrainedTokenizerFast]:
    @pytest.fixture(scope="session")
    def fixture() -> PreTrainedTokenizerFast:
        from eh30pt.modeling.tokenizer import load_tokenizer

        return load_tokenizer(path)

    return fixture


def _make_dataset_fixture(
    dataset: DatasetName,
    model: str,
) -> Callable[[pytest.FixtureRequest], DataSplits]:
    @pytest.fixture(scope="session")
    def fixture(request: pytest.FixtureRequest) -> DataSplits:
        from datasets.load import load_dataset
        from datasets.utils.info_utils import VerificationMode

        from eh30pt.datasets.multinerd import Multinerd
        from eh30pt.datasets.oblique import load_obl
        from eh30pt.modeling.arch import infer_architecture
        from eh30pt.types import DataSplits

        config: PreTrainedConfig = request.getfixturevalue(f"{model}_config")
        tokenizer: PreTrainedTokenizerFast = request.getfixturevalue(
            f"{model}_tokenizer"
        )

        arch = infer_architecture(config)
        if dataset == "oblique":
            return load_obl(tokenizer, arch)

        split = {
            "train": "train[:64]",
            "validation": "validation[:64]",
            "test": "test[:64]",
        }

        raw = load_dataset(
            "Babelscape/multinerd",
            split=cast("Split", split),
            verification_mode=VerificationMode.NO_CHECKS,
        )

        raw = cast("DatasetDict", raw)
        return DataSplits(
            Multinerd(raw["train"], tokenizer, arch),
            Multinerd(raw["validation"], tokenizer, arch),
            Multinerd(raw["test"], tokenizer, arch),
        )

    return fixture


def _make_model_fixture(
    dataset: DatasetName,
    model: str,
    path: str,
) -> Callable[[pytest.FixtureRequest], PreTrainedModel]:
    @pytest.fixture(scope="session")
    def fixture(request: pytest.FixtureRequest) -> PreTrainedModel:
        from eh30pt.modeling.arch import infer_architecture
        from eh30pt.modeling.loader import load_model

        config: PreTrainedConfig = request.getfixturevalue(f"{model}_config")
        tokenizer: PreTrainedTokenizerFast = request.getfixturevalue(
            f"{model}_tokenizer"
        )

        arch = infer_architecture(config)

        return load_model(tokenizer, path, dataset, arch, head_only=False)

    return fixture


for _model, _path in models.items():
    globals()[f"{_model}_config"] = _make_config_fixture(_path)
    globals()[f"{_model}_tokenizer"] = _make_tokenizer_fixture(_path)

    for _dataset in _datasets:
        globals()[f"{_dataset}_{_model}_dataset"] = _make_dataset_fixture(
            _dataset,
            _model,
        )

        globals()[f"{_dataset}_{_model}_model"] = _make_model_fixture(
            _dataset,
            _model,
            _path,
        )

_matrix = [(dataset, model) for dataset in _datasets for model in models]
_matrix_ids = [f"{dataset}-{model}" for dataset, model in _matrix]


@pytest.fixture(params=_matrix, scope="session", ids=_matrix_ids)
def model_setup(request: pytest.FixtureRequest) -> ModelSetup:
    from eh30pt.datasets.registry import get_labels

    dataset, model = request.param
    return ModelSetup(
        request.getfixturevalue(f"{model}_config"),
        request.getfixturevalue(f"{model}_tokenizer"),
        get_labels(dataset),
        request.getfixturevalue(f"{dataset}_{model}_dataset"),
        request.getfixturevalue(f"{dataset}_{model}_model"),
    )


@pytest.fixture(scope="session")
def database() -> Iterator[None]:
    """Seeded postgres for Est-Health tests, exposed through PostgresSettings."""
    import os
    from pathlib import Path
    from urllib.parse import unquote, urlsplit

    from sqlalchemy import create_engine, text
    from testcontainers.community.postgres import PostgresContainer

    seed_dir = Path(__file__).parents[2] / "sql"
    statements: list[str] = [
        part.strip()
        for seed in sorted(seed_dir.glob("*.sql"))
        for part in seed.read_text(encoding="utf-8").split(";")
        if part.strip()
    ]

    env_vars = (
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "TESTCONTAINERS_RYUK_DISABLED",
    )
    saved: dict[str, str | None] = {var: os.environ.get(var) for var in env_vars}
    os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"

    with PostgresContainer(image="postgres:17", driver="psycopg") as postgres:
        engine = create_engine(postgres.get_connection_url())

        with engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))

        engine.dispose()

        parts = urlsplit(postgres.get_connection_url())
        os.environ["POSTGRES_HOST"] = parts.hostname or "localhost"
        os.environ["POSTGRES_PORT"] = str(parts.port or 5432)
        os.environ["POSTGRES_DB"] = unquote(parts.path.lstrip("/"))
        os.environ["POSTGRES_USER"] = unquote(parts.username or "")
        os.environ["POSTGRES_PASSWORD"] = unquote(parts.password or "")

        try:
            yield
        finally:
            for var in env_vars:
                value = saved[var]
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
