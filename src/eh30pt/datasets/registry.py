"""Dataset registry: map dataset names to streaming loaders."""

from typing import TYPE_CHECKING

from eh30pt.datasets.discontinuation.classifier import (
    DiscontClassifier,
    load_discont_classifier,
)
from eh30pt.datasets.discontinuation.decider import (
    DiscontDecider,
    load_discont_decider,
)
from eh30pt.datasets.discontinuation.extract import (
    DiscontExtract,
    load_discont_extract,
)
from eh30pt.datasets.multinerd import Multinerd, load_multinerd
from eh30pt.datasets.oblique import Obl, load_obl

if TYPE_CHECKING:
    from openai.types.shared_params import ResponseFormatJSONSchema
    from transformers import PreTrainedTokenizerFast

    from eh30pt.types import (
        Architecture,
        DatasetName,
        DatasetTask,
        DataSource,
        DataSplits,
        ExtractShot,
        FewShot,
    )


def get_task(dataset: DatasetName) -> DatasetTask:
    """Return the task of `dataset` without loading data."""
    if dataset == "multinerd":
        return Multinerd.task

    if dataset == "oblique":
        return Obl.task

    if dataset == "discont-classifier":
        return DiscontClassifier.task

    if dataset == "discont-decider":
        return DiscontDecider.task

    if dataset == "discont-extract":
        return DiscontExtract.task

    msg = f"invalid dataset {dataset}"
    raise ValueError(msg)


def assert_classification(dataset: DatasetName, action: str) -> None:
    """Raise ValueError when `dataset` is an extraction dataset."""
    if get_task(dataset) == "extraction":
        msg = f"dataset '{dataset}' is an extraction dataset, {action}"
        raise ValueError(msg)


def assert_extraction(dataset: DatasetName, action: str) -> None:
    """Raise ValueError when `dataset` is a classification dataset."""
    if get_task(dataset) == "classification":
        msg = f"dataset '{dataset}' is a classification dataset, {action}"
        raise ValueError(msg)


def get_labels(dataset: DatasetName) -> tuple[str, ...]:
    """Return the label values of `dataset` in id order."""
    if dataset == "discont-extract":
        msg = f"dataset '{dataset}' is an extraction dataset and has no labels"
        raise ValueError(msg)

    if dataset == "multinerd":
        return tuple(Multinerd.id2label.values())

    if dataset == "oblique":
        return tuple(Obl.id2label.values())

    if dataset == "discont-classifier":
        return tuple(DiscontClassifier.id2label.values())

    if dataset == "discont-decider":
        return tuple(DiscontDecider.id2label.values())

    msg = f"invalid dataset {dataset}"
    raise ValueError(msg)


def get_sys_prompt(dataset: DatasetName) -> str:
    """Return the system prompt for `dataset` without loading data."""
    if dataset == "multinerd":
        return Multinerd.system_prompt

    if dataset == "oblique":
        return Obl.system_prompt

    if dataset == "discont-classifier":
        return DiscontClassifier.system_prompt

    if dataset == "discont-decider":
        return DiscontDecider.system_prompt

    if dataset == "discont-extract":
        return DiscontExtract.system_prompt

    msg = f"invalid dataset {dataset}"
    raise ValueError(msg)


def get_response_schema(dataset: DatasetName) -> ResponseFormatJSONSchema:
    """Return the structured generation schema for `dataset` without loading data."""
    if dataset == "multinerd":
        return Multinerd.response_format()

    if dataset == "oblique":
        return Obl.response_format()

    if dataset == "discont-classifier":
        return DiscontClassifier.response_format()

    if dataset == "discont-decider":
        return DiscontDecider.response_format()

    if dataset == "discont-extract":
        return DiscontExtract.response_format()

    msg = f"invalid dataset {dataset}"
    raise ValueError(msg)


def get_few_shots(dataset: DatasetName, n_shot: int) -> list[FewShot]:
    """Return the first `n_shot` hardcoded examples for `dataset`."""
    if dataset == "multinerd":
        shots = Multinerd.few_shots
    elif dataset == "discont-classifier":
        shots = DiscontClassifier.few_shots
    elif dataset == "discont-decider":
        shots = DiscontDecider.few_shots
    elif dataset == "discont-extract":
        shots = DiscontExtract.few_shots
    elif dataset == "oblique":
        shots = Obl.few_shots
    else:
        msg = f"invalid dataset {dataset}"
        raise ValueError(msg)

    if n_shot > len(shots):
        msg = f"requested {n_shot} few shot examples, only {len(shots)} defined"
        raise ValueError(msg)

    return shots[:n_shot]


def get_extract_shots(dataset: DatasetName, n_shot: int) -> list[ExtractShot]:
    """Return the first `n_shot` extraction examples for `dataset`."""
    if dataset == "discont-extract":
        shots = DiscontExtract.extract_shots
    else:
        msg = f"dataset '{dataset}' has no extraction examples"
        raise ValueError(msg)

    if n_shot > len(shots):
        msg = f"requested {n_shot} extraction examples, only {len(shots)} defined"
        raise ValueError(msg)

    return shots[:n_shot]


def get_extract_fields(dataset: DatasetName) -> tuple[str, ...]:
    """Return the required extraction fields of `dataset` from its schema."""
    if get_task(dataset) == "classification":
        msg = (
            f"dataset '{dataset}' is a classification dataset "
            "and has no extraction fields"
        )
        raise ValueError(msg)

    required = get_response_schema(dataset)["json_schema"]["schema"].get("required", [])
    if not isinstance(required, list) or not all(
        isinstance(field, str) for field in required
    ):
        msg = f"dataset '{dataset}' schema has no required extraction fields"
        raise ValueError(msg)

    return tuple(field for field in required if isinstance(field, str))


def load_data(
    tokenizer: PreTrainedTokenizerFast,
    dataset: DatasetName,
    arch: Architecture,
    *,
    n_shot: int = 0,
    n_virtual: int = 0,
    start_after: int | None = None,
    source: DataSource = "trigger",
) -> DataSplits:
    """Load dataset splits, streaming only rows with id above start_after."""
    _with_offset = ("discont-classifier", "discont-decider", "discont-extract")
    if start_after is not None and dataset not in _with_offset:
        msg = f"start_after is only supported for discont datasets, got {dataset}"
        raise ValueError(msg)

    if source != "trigger" and dataset not in _with_offset:
        msg = f"source '{source}' is only supported for discont datasets, got {dataset}"
        raise ValueError(msg)

    if dataset == "multinerd":
        return load_multinerd(tokenizer, arch, n_shot, n_virtual)

    if dataset == "oblique":
        return load_obl(tokenizer, arch, n_shot, n_virtual)

    if dataset == "discont-classifier":
        return load_discont_classifier(
            tokenizer,
            arch,
            n_shot,
            n_virtual,
            source=source,
            start_after=start_after,
        )

    if dataset == "discont-decider":
        return load_discont_decider(
            tokenizer,
            arch,
            n_shot,
            n_virtual,
            source=source,
            start_after=start_after,
        )

    if dataset == "discont-extract":
        return load_discont_extract(
            tokenizer,
            arch,
            n_shot,
            n_virtual,
            source=source,
            start_after=start_after,
        )

    msg = f"invalid dataset {dataset}"
    raise ValueError(msg)
