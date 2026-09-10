"""Tests for environment settings."""

from typing import TYPE_CHECKING, Final

import pytest
from pydantic import ValidationError

from eh30pt.settings import (
    CommonSettings,
    LoRASettings,
    MlflowSettings,
    PostgresSettings,
    ServerSettings,
    StreamSettings,
    TrainingSettings,
)

if TYPE_CHECKING:
    from pathlib import Path

_port_default: Final = 5432
_port_custom: Final = 5433
_tolerance_default: Final = 0.01
_train_steps_custom: Final = 10
_batch_size_default: Final = 8
_learning_rate_custom: Final = 5e-5
_warmup_ratio_default: Final = 0.06
_lora_rank_default: Final = 16
_lora_alpha_default: Final = 16
_lora_dropout_default: Final = 0.05
_concurrency_default: Final = 8
_shard_size_default: Final = 10_000
_log_interval_default: Final = 1000


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for var in (
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_defaults() -> None:
    settings = PostgresSettings()

    assert settings.host == "localhost"
    assert settings.port == _port_default
    assert settings.db == "postgres"
    assert settings.user == "postgres"
    assert settings.password.get_secret_value() == "postgres"
    assert (
        settings.connection_url
        == "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
    )


def test_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", str(_port_custom))
    monkeypatch.setenv("POSTGRES_DB", "prod")
    monkeypatch.setenv("POSTGRES_USER", "app")
    monkeypatch.setenv("POSTGRES_PASSWORD", "s3cret")

    settings = PostgresSettings()

    assert settings.host == "db.internal"
    assert settings.port == _port_custom
    assert settings.db == "prod"
    assert settings.user == "app"
    assert settings.password.get_secret_value() == "s3cret"


def test_connection_url_encodes_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss:word/!")

    settings = PostgresSettings()

    assert settings.connection_url == (
        "postgresql+psycopg://postgres:p%40ss%3Aword%2F%21@localhost:5432/postgres"
    )


def test_mlflow_defaults() -> None:
    settings = MlflowSettings()

    assert settings.tracking_uri == "sqlite:///mlflow.db"
    assert settings.experiment == "eh30pt"
    assert settings.run_name is None
    assert settings.run_id is None


def test_mlflow_explicit_values() -> None:
    settings = MlflowSettings(
        tracking_uri="http://mlflow:5000",
        experiment="prod",
        run_name="run-a",
        run_id="abc123",
    )

    assert settings.tracking_uri == "http://mlflow:5000"
    assert settings.experiment == "prod"
    assert settings.run_name == "run-a"
    assert settings.run_id == "abc123"


def test_mlflow_artifact_paths() -> None:
    assert MlflowSettings.model_path == "model"
    assert MlflowSettings.predictions_path == "predictions"


def test_training_defaults() -> None:
    settings = TrainingSettings(
        train_steps=_train_steps_custom, learning_rate=_learning_rate_custom
    )

    assert settings.eval_steps == 0
    assert settings.patience == 0
    assert settings.tolerance == _tolerance_default
    assert settings.train_steps == _train_steps_custom
    assert settings.batch_size == _batch_size_default
    assert settings.learning_rate == _learning_rate_custom
    assert settings.warmup_ratio == _warmup_ratio_default


def test_lora_defaults() -> None:
    settings = LoRASettings()

    assert settings.rank == _lora_rank_default
    assert settings.alpha == _lora_alpha_default
    assert settings.dropout == _lora_dropout_default


def test_stream_defaults() -> None:
    settings = StreamSettings()

    assert settings.concurrency == _concurrency_default
    assert settings.shard_size == _shard_size_default
    assert settings.log_interval == _log_interval_default
    assert settings.from_id is None


def test_stream_rejects_non_positive() -> None:
    with pytest.raises(ValidationError, match="concurrency"):
        StreamSettings(concurrency=0)

    with pytest.raises(ValidationError, match="shard_size"):
        StreamSettings(shard_size=0)

    with pytest.raises(ValidationError, match="log_interval"):
        StreamSettings(log_interval=-1)

    with pytest.raises(ValidationError, match="from_id"):
        StreamSettings(from_id=-1)


def test_mlflow_rejects_empty() -> None:
    with pytest.raises(ValidationError, match="tracking_uri"):
        MlflowSettings(tracking_uri="")

    with pytest.raises(ValidationError, match="experiment"):
        MlflowSettings(experiment="")


def test_training_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError, match="train_steps"):
        TrainingSettings(train_steps=0, learning_rate=_learning_rate_custom)

    with pytest.raises(ValidationError, match="batch_size"):
        TrainingSettings(
            train_steps=_train_steps_custom,
            batch_size=0,
            learning_rate=_learning_rate_custom,
        )

    with pytest.raises(ValidationError, match="learning_rate"):
        TrainingSettings(train_steps=_train_steps_custom, learning_rate=-1.0)

    with pytest.raises(ValidationError, match="eval_steps"):
        TrainingSettings(
            train_steps=_train_steps_custom,
            learning_rate=_learning_rate_custom,
            eval_steps=-1,
        )


def test_lora_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError, match="rank"):
        LoRASettings(rank=0)

    with pytest.raises(ValidationError, match="dropout"):
        LoRASettings(dropout=1.5)


def test_server_defaults() -> None:
    settings = ServerSettings()

    assert settings.provider == "llama"
    assert settings.api_key == "no-auth"
    assert settings.base_url == "http://localhost:8080"


def test_server_resolves_provider_default() -> None:
    assert ServerSettings(provider="vllm").base_url == "http://localhost:8000"


def test_common_defaults() -> None:
    settings = CommonSettings(model="some-model", dataset="oblique")

    assert settings.model == "some-model"
    assert settings.dataset == "oblique"
    assert settings.arch is None
    assert settings.n_shot == 0


def test_common_rejects_invalid() -> None:
    with pytest.raises(ValidationError, match="dataset"):
        CommonSettings(
            model="some-model",
            dataset="unknown",  # ty: ignore[invalid-argument-type]
        )

    with pytest.raises(ValidationError, match="n_shot"):
        CommonSettings(model="some-model", dataset="oblique", n_shot=-1)
