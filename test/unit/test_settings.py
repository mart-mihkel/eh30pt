"""Tests for environment settings."""

from typing import TYPE_CHECKING, Final

import pytest
from pydantic import ValidationError

from eh30pt.settings import (
    CommonSettings,
    LabelStudioSettings,
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
_epochs_custom: Final = 3.0
_train_size_custom: Final = 100
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
        "LABEL_STUDIO_URL",
        "LABEL_STUDIO_TOKEN",
        "LABEL_STUDIO_AUTH",
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


def test_mlflow_rejects_empty() -> None:
    with pytest.raises(ValidationError, match="tracking_uri"):
        MlflowSettings(tracking_uri="")

    with pytest.raises(ValidationError, match="experiment"):
        MlflowSettings(experiment="")


def test_training_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError, match="epochs"):
        TrainingSettings(epochs=0, learning_rate=_learning_rate_custom)

    with pytest.raises(ValidationError, match="train_size"):
        TrainingSettings(
            epochs=_epochs_custom,
            train_size=0,
            learning_rate=_learning_rate_custom,
        )

    with pytest.raises(ValidationError, match="batch_size"):
        TrainingSettings(
            epochs=_epochs_custom,
            batch_size=0,
            learning_rate=_learning_rate_custom,
        )

    with pytest.raises(ValidationError, match="learning_rate"):
        TrainingSettings(epochs=_epochs_custom, learning_rate=-1.0)

    with pytest.raises(ValidationError, match="eval_steps"):
        TrainingSettings(
            epochs=_epochs_custom,
            learning_rate=_learning_rate_custom,
            eval_steps=-1,
        )


def test_training_stores_epochs_and_train_size() -> None:
    settings = TrainingSettings(
        epochs=_epochs_custom,
        train_size=_train_size_custom,
        learning_rate=_learning_rate_custom,
    )

    assert settings.epochs == _epochs_custom
    assert settings.train_size == _train_size_custom
    assert settings.save_model is True
    assert (
        TrainingSettings(
            epochs=_epochs_custom, learning_rate=_learning_rate_custom
        ).train_size
        is None
    )


def test_training_disables_model_saving() -> None:
    settings = TrainingSettings(
        epochs=_epochs_custom,
        learning_rate=_learning_rate_custom,
        save_model=False,
    )

    assert settings.save_model is False


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
        )

    with pytest.raises(ValidationError, match="n_shot"):
        CommonSettings(model="some-model", dataset="oblique", n_shot=-1)


def test_label_studio_defaults_to_the_development_token() -> None:
    settings = LabelStudioSettings()

    assert settings.url == "http://localhost:8080"
    assert settings.token.get_secret_value() == "eh30pt-dev-token"
    assert settings.auth == "legacy"


def test_label_studio_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LABEL_STUDIO_URL", "https://label.internal")
    monkeypatch.setenv("LABEL_STUDIO_TOKEN", "s3cret")
    monkeypatch.setenv("LABEL_STUDIO_AUTH", "refresh")

    settings = LabelStudioSettings()

    assert settings.url == "https://label.internal"
    assert settings.token.get_secret_value() == "s3cret"
    assert settings.auth == "refresh"


def test_label_studio_rejects_unknown_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LABEL_STUDIO_AUTH", "bearer")

    with pytest.raises(ValidationError, match="auth"):
        LabelStudioSettings()
