"""Environment settings."""

from typing import ClassVar
from urllib.parse import quote

from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from eh30pt.types import Architecture, DatasetName, Provider


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


class MlflowSettings(BaseModel):
    """MLflow tracking settings."""

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

    eval_steps: int = Field(default=0, ge=0)
    patience: int = Field(default=0, ge=0)
    tolerance: float = Field(default=0.01, ge=0)
    train_steps: int = Field(ge=1)
    batch_size: int = Field(default=8, ge=1)
    learning_rate: float = Field(ge=0)
    warmup_ratio: float = Field(default=0.06, ge=0)


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
    from_id: int | None = Field(default=None, ge=0)


class ServerSettings(BaseModel):
    """Server connection settings."""

    provider: Provider = "llama"
    api_key: str = "no-auth"
    timeout: int = Field(default=300, ge=0)
    format: ResponseFormatJSONSchema | None = None

    @property
    def base_url(self) -> str:
        """Return the explicit base URL, falling back to the provider default."""
        if self.provider == "vllm":
            return "http://localhost:8000"

        if self.provider == "llama":
            return "http://localhost:9931"

        msg = f"invalid provider {self.provider}"
        raise ValueError(msg)


class CommonSettings(BaseModel):
    """Model and data selection."""

    model: str
    dataset: DatasetName
    arch: Architecture | None = None
    n_shot: int = Field(default=0, ge=0)
