"""Environment settings."""

from pathlib import Path
from typing import ClassVar, Literal
from urllib.parse import quote

from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from eh30pt.types import Architecture, DatasetName, EvalSteps, Provider


class PostgresSettings(BaseSettings):
    """Postgres connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="postgres_",
        env_file=".env",
        extra="ignore",
    )

    port: int = 5432
    host: str = "localhost"

    db: str = "postgres"
    user: str = "postgres"
    password: SecretStr = SecretStr("postgres")

    @property
    def connection_url(self) -> str:
        """Build a SQLAlchemy postgres connection URL."""
        proto = "postgresql+psycopg"
        user = quote(self.user, safe="")
        password = quote(self.password.get_secret_value(), safe="")
        return f"{proto}://{user}:{password}@{self.host}:{self.port}/{self.db}"


class MlflowSettings(BaseSettings):
    """MLflow tracking settings."""

    model_config = SettingsConfigDict(
        env_prefix="mlflow_",
        env_file=".env",
        extra="ignore",
    )

    model_path: ClassVar[str] = "model"
    """Model artifact path."""

    predictions_path: ClassVar[str] = "predictions"
    """Prediction artifact path."""

    tracking_uri: str = Field(default="sqlite:///mlflow.db", min_length=1)
    experiment: str = Field(default="eh30pt", min_length=1)
    run_name: str | None = None
    run_id: str | None = None


class TrainingSettings(BaseModel):
    """Shared training loop hyperparameters."""

    eval_steps: EvalSteps = Field(default=0)
    patience: int = Field(default=0, ge=0)
    tolerance: float = Field(default=0.01, ge=0)
    epochs: float = Field(gt=0)
    train_size: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=8, ge=1)
    learning_rate: float = Field(ge=0)
    warmup_ratio: float = Field(default=0.06, ge=0)
    save_model: bool = False


class LoRASettings(BaseModel):
    """LoRA adapter hyperparameters."""

    rank: int = Field(default=16, ge=1)
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.05, ge=0, le=1)


class StreamSettings(BaseModel):
    """Server streaming hyperparameters."""

    concurrency: int = Field(default=8, ge=1)
    shard_size: int = Field(default=10_000, ge=1)
    log_interval: int = Field(default=1000, ge=1)


class LabelStudioSettings(BaseSettings):
    """Label Studio API connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="label_studio_",
        env_file=".env",
        extra="ignore",
    )

    url: str = "http://localhost:8080"
    token: SecretStr = SecretStr("eh30pt-dev-token")
    """API token, defaulting to the one compose pins on the dev user."""

    auth: Literal["legacy", "refresh"] = "legacy"
    """
    How `token` authenticates.

    A legacy token travels as `Authorization: Token <token>` and is looked up
    as the user token; a refresh token is exchanged for a short lived access
    token sent as `Authorization: Bearer <access>`.
    """


class ServerSettings(BaseSettings):
    """Server connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="openai_",
        env_file=".env",
        extra="ignore",
    )

    provider: Provider = "llama"
    api_key: str = "no-auth"
    timeout: int = Field(default=300, ge=0)
    format: ResponseFormatJSONSchema | None = None
    port: int | None = Field(default=None, ge=1, le=65535)

    @property
    def base_url(self) -> str:
        """Return the local server URL for the selected provider and port."""
        if self.provider == "vllm":
            port = self.port or 8000
        elif self.provider == "llama":
            port = self.port or 9931
        else:
            msg = f"invalid provider {self.provider}"
            raise ValueError(msg)

        return f"http://localhost:{port}"


class CommonSettings(BaseModel):
    """Model and data selection."""

    model: str
    dataset: DatasetName
    arch: Architecture | None = None
    sequence_classification: bool = False
    n_shot: int = Field(default=0, ge=0)
    system_prompt: str | None = None
    data_path: Path | None = None
    seed: int | None = None
