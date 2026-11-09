"""Types and stubs."""

from collections.abc import Callable
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    NamedTuple,
    NotRequired,
    Protocol,
    TypedDict,
)

from pydantic import Field
from returns.result import Failure, Result, Success

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from torch.utils.data import IterableDataset as TorchIterableDataset
    from transformers import EvalPrediction
    from ty_extensions import Intersection

    from eh30pt.datasets.mixin import DatasetMixin

type LogLevel = Literal[
    "debug",
    "info",
    "warning",
    "error",
]

type Architecture = Literal[
    "encoder",
    "decoder",
    "encoder-decoder",
]

type DatasetName = Literal[
    "multinerd",
    "oblique",
    "discont-classify/extract-labels",
    "discont-binary/extract-labels",
    "discont-extract/trigger",
    "discont-extract/observed",
    "discont-extract/discont",
    "discont-extract/switch",
    "discont-extract/usage-gap",
]

type Method = Literal[
    "fine-tune",
    "cls-head",
    "prompt-tune",
    "qlora",
    "few-shot",
    "systemone",
    "langextract",
    "silver-labels",
]

type DatasetTask = Literal[
    "classification",
    "extraction",
]

type DgEventType = Literal[
    "discontinuation",
    "switch",
    "usage_gap",
]

type DiscontClassifierLabel = Literal[
    "Adverse reactions",
    "Treatment success",
    "Treatment inefficacy",
    "Contraindication/unsafe co-usage",
    "Other",
    "Adverse reaction without discontinuation",
    "Treatment inefficacy without discontinuation",
    "Dosage change",
    "Other non-discontinuation",
    "Incomplete phrase",
    "Incorrect phrase",
]

type DataSource = Literal[
    "observed",
    "trigger",
    "discont",
    "switch",
    "usage-gap",
]

type PipelineTask = Literal[
    "text-classification",
    "text-generation",
]

type Provider = Literal[
    "llama",
    "vllm",
]

type PrefixInit = Literal[
    "pretrained",
    "random",
]

type EvalSteps = Annotated[int, Field(ge=0)] | Literal["epoch"]

type ComputeMetricsFn = Callable[[EvalPrediction], dict[str, float]]


class RunConfig(TypedDict):
    """Full run configuration: run identity plus optional provenance."""

    method: Method
    dataset: DatasetName
    architecture: Architecture
    base_model: str
    n_shot: int
    batch_size: int
    sequence_classification: NotRequired[bool]
    system_prompt: NotRequired[str]
    total_parameters: NotRequired[int]
    trainable_parameters: NotRequired[int]


class TrainingRunConfig(RunConfig):
    """Run configuration shared by the training scripts."""

    eval_steps: EvalSteps
    patience: int
    tolerance: float
    epochs: float
    train_size: int | None
    learning_rate: float
    warmup_ratio: float
    save_model: bool


class FineTuneRunConfig(TrainingRunConfig):
    """Run configuration for fine-tuning."""

    head_only: bool


class PromptTuneRunConfig(TrainingRunConfig):
    """Run configuration for prompt-tuning."""

    prefix_init: PrefixInit
    n_virtual: int


class QLoRaRunConfig(TrainingRunConfig):
    """Run configuration for QLoRA tuning."""

    lora_r: int
    lora_alpha: int
    lora_dropout: float


class FewShotRunConfig(RunConfig):
    """Run configuration for few-shot evaluation."""


class FewShot(NamedTuple):
    """User prompt paired with its gold label and optional dataset row id."""

    user: str
    label: str
    id: int | None = None


class ExtractShot(NamedTuple):
    """Example extraction pairing input text with field spans."""

    user: str
    extractions: dict[str, str]


class GenerateRecord(NamedTuple):
    """One llama prediction row keyed by dataset row id."""

    id: int
    user: str
    prediction: str


class EvalRecord(NamedTuple):
    """One test eval row pairing the gold label with its prediction."""

    id: int | None
    user: str
    label: str
    prediction: str


class DataSplits(NamedTuple):
    """Per-split streaming datasets for training."""

    train: Intersection[TorchIterableDataset, DatasetMixin]
    eval: Intersection[TorchIterableDataset, DatasetMixin]
    test: Intersection[TorchIterableDataset, DatasetMixin]


class ArtifactLogger(Protocol):
    """Minimal run context surface needed to persist prediction shards."""

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """Log a local file as a run artifact."""


class Predictor[ItemT, ResultT](Protocol):
    """Async single-item predictor for bounded-concurrency streaming."""

    async def predict(self, item: ItemT, /) -> ResultT:
        """Predict one result for item."""


def asafe[**P, T](
    fn: Callable[P, Awaitable[T]],
) -> Callable[P, Awaitable[Result[T, Exception]]]:
    """Make async function safe."""

    @wraps(fn)
    async def wrapper(
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Result[T, Exception]:
        try:
            return Success(await fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            return Failure(exc)

    return wrapper
