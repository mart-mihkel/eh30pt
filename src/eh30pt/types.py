"""Types and stubs."""

from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from datasets.dataset_dict import DatasetDict
    from datasets.splits import Split
    from transformers import EvalPrediction

type LogLevel = Literal["debug", "info", "warning", "error"]
type Architecture = Literal["encoder", "decoder", "encoder-decoder"]
type DatasetName = Literal["multinerd", "oblique"]
type PrefixInit = Literal["pretrained", "random"]
type ComputeMetricsFn = Callable[[EvalPrediction], dict[str, float]]


class RunConfig(TypedDict):
    """Full run configuration: resume key, logged params, and identity."""

    method: str
    dataset: DatasetName
    architecture: Architecture
    base_model: str
    n_shot: int
    batch_size: int
    seed: int | None
    system_prompt: str
    train_samples: int
    validation_samples: int
    test_samples: int
    total_parameters: int
    trainable_parameters: int


class TrainingRunConfig(RunConfig):
    """Run configuration shared by the training scripts."""

    train_subset: int | None
    val_subset: int | None
    do_eval: bool
    patience: int
    tolerance: float
    epochs: int
    learning_rate: float
    warmup_ratio: float


class FineTuneRunConfig(TrainingRunConfig):
    """Run configuration for fine-tuning."""

    head_only: bool


class PromptTuneRunConfig(TrainingRunConfig):
    """Run configuration for prompt-tuning."""

    prefix_init: PrefixInit
    n_virtual: int


class FewShotRunConfig(RunConfig):
    """Run configuration for few-shot evaluation."""


@runtime_checkable
class DatasetProtocol(Protocol):
    """Common trait implemented by dataset loaders."""

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the configured architecture."""
        ...

    @property
    def id2label(self) -> Mapping[int, str]:
        """Return the label id to label name mapping."""
        ...

    @property
    def label2id(self) -> Mapping[str, int]:
        """Return the label name to label id mapping."""
        ...

    @property
    def data(self) -> DatasetDict:
        """Return the loaded dataset splits."""
        ...

    def load(
        self,
        *,
        n_shot: int = 0,
        n_virtual: int = 0,
        split: Split | None = None,
    ) -> None:
        """Load the dataset into instance state."""
        ...
