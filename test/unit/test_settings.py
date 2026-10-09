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

_port_custom: Final = 5433
_train_steps_custom: Final = 10
_learning_rate_custom: Final = 5e-5


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
        "MLFLOW_TRACKING_URI",
        "LABEL_STUDIO_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


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


def test_server_resolves_provider_default() -> None:
    assert ServerSettings(provider="vllm").base_url == "http://localhost:8000"


def test_server_rejects_invalid_provider() -> None:
    settings = ServerSettings(provider="llama")
    settings.provider = "unknown"  # ty: ignore[invalid-assignment]

    with pytest.raises(ValueError, match="invalid provider"):
        _ = settings.base_url


def test_common_rejects_invalid() -> None:
    with pytest.raises(ValidationError, match="dataset"):
        CommonSettings(
            model="some-model",
            dataset="unknown",  # ty: ignore[invalid-argument-type]
            source="trigger",
        )

    with pytest.raises(ValidationError, match="n_shot"):
        CommonSettings(
            model="some-model", dataset="oblique", n_shot=-1, source="trigger"
        )
