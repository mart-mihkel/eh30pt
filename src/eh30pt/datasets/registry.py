"""Dataset registry: map dataset names to streaming loaders."""

from typing import TYPE_CHECKING

from eh30pt.datasets.discontinuation.binary import DiscontBinary, load_discont_binary
from eh30pt.datasets.discontinuation.classify import (
    DiscontClassifier,
    load_discont_classifier,
)
from eh30pt.datasets.discontinuation.extract import (
    DiscontExtract,
    load_discont_extract,
)
from eh30pt.datasets.multinerd import Multinerd, load_multinerd
from eh30pt.datasets.oblique import Obl, load_obl

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def _is_discont(dataset: DatasetName) -> bool:
    return dataset.startswith("discont-")


def _is_classifier(dataset: DatasetName) -> bool:
    return dataset.startswith(("discont-classifier", "discont-binary/extract-labels"))


def _is_extract(dataset: DatasetName) -> bool:
    return dataset.startswith("discont-extract/")


def _extract_source(dataset: DatasetName) -> DataSource:
    match dataset.removeprefix("discont-extract/"):
        case "trigger" | "observed" | "discont" | "switch" | "usage-gap" as source:
            return source

    msg = f"dataset '{dataset}' names no Est-Health query"
    raise ValueError(msg)


def get_task(dataset: DatasetName) -> DatasetTask:
    """Return the task of `dataset` without loading data."""
    if dataset == "multinerd":
        return Multinerd.task

    if dataset == "oblique":
        return Obl.task

    if _is_classifier(dataset):
        return DiscontClassifier.task

    if _is_extract(dataset):
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
    if dataset == "multinerd":
        return tuple(Multinerd.id2label.values())

    if dataset == "oblique":
        return tuple(Obl.id2label.values())

    if dataset == "discont-classifier/extract-labels":
        return tuple(DiscontClassifier.id2label.values())

    if dataset == "discont-binary/extract-labels":
        return tuple(DiscontBinary.id2label.values())

    if _is_extract(dataset):
        msg = f"dataset '{dataset}' is an extraction dataset and has no labels"
        raise ValueError(msg)

    msg = f"invalid dataset {dataset}"
    raise ValueError(msg)


def get_sys_prompt(dataset: DatasetName) -> str:
    """Return the system prompt for `dataset` without loading data."""
    if dataset == "multinerd":
        return Multinerd.system_prompt

    if dataset == "oblique":
        return Obl.system_prompt

    if dataset == "discont-classifier/extract-labels":
        return DiscontClassifier.system_prompt

    if dataset == "discont-binary/extract-labels":
        return DiscontBinary.system_prompt

    if _is_extract(dataset):
        return DiscontExtract.system_prompt

    msg = f"invalid dataset {dataset}"
    raise ValueError(msg)


def get_response_schema(dataset: DatasetName) -> ResponseFormatJSONSchema:
    """Return the structured generation schema for `dataset` without loading data."""
    if dataset == "multinerd":
        return Multinerd.response_format()

    if dataset == "oblique":
        return Obl.response_format()

    if dataset == "discont-classifier/extract-labels":
        return DiscontClassifier.response_format()

    if dataset == "discont-binary/extract-labels":
        return DiscontBinary.response_format()

    if _is_extract(dataset):
        return DiscontExtract.response_format()

    msg = f"invalid dataset {dataset}"
    raise ValueError(msg)


def get_few_shots(dataset: DatasetName, n_shot: int) -> list[FewShot]:
    """Return the first `n_shot` hardcoded examples for `dataset`."""
    if dataset == "multinerd":
        shots = Multinerd.few_shots
    elif dataset == "discont-classifier/extract-labels":
        shots = DiscontClassifier.few_shots
    elif dataset == "discont-binary/extract-labels":
        shots = DiscontBinary.few_shots
    elif _is_extract(dataset):
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
    if not _is_extract(dataset):
        msg = f"dataset '{dataset}' has no extraction examples"
        raise ValueError(msg)

    shots = DiscontExtract.extract_shots
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


def _check_splits(
    dataset: DatasetName,
    start_after: int | None,
    labeling_projects: Sequence[int] | None,
) -> None:
    """Raise when the split options do not fit the dataset."""
    if start_after is not None and not _is_discont(dataset):
        msg = f"start_after is only supported for discont datasets, got {dataset}"
        raise ValueError(msg)

    if not _is_classifier(dataset) and labeling_projects is not None:
        msg = f"project ids are only read by the discont classifiers, got '{dataset}'"
        raise ValueError(msg)


def load_data(
    tokenizer: PreTrainedTokenizerFast,
    dataset: DatasetName,
    arch: Architecture,
    *,
    n_shot: int = 0,
    n_virtual: int = 0,
    start_after: int | None = None,
    labeling_projects: Sequence[int] | None = None,
) -> DataSplits:
    """Load dataset splits, streaming only rows with id above start_after."""
    _check_splits(dataset, start_after, labeling_projects)

    if dataset == "multinerd":
        return load_multinerd(tokenizer, arch, n_shot, n_virtual)

    if dataset == "oblique":
        return load_obl(tokenizer, arch, n_shot, n_virtual)

    if _is_extract(dataset):
        return load_discont_extract(
            tokenizer,
            arch,
            n_shot,
            n_virtual,
            source=_extract_source(dataset),
            start_after=start_after,
        )

    if _is_classifier(dataset):
        if labeling_projects is None:
            msg = f"dataset '{dataset}' needs project ids"
            raise ValueError(msg)

        if dataset == "discont-classifier/extract-labels":
            load_discont_classifier(
                tokenizer,
                arch,
                n_shot,
                n_virtual,
                start_after=start_after,
                labeling_projects=labeling_projects,
            )

        if dataset == "discont-binary/extract-labels":
            return load_discont_binary(
                tokenizer,
                arch,
                n_shot,
                n_virtual,
                start_after=start_after,
                labeling_projects=labeling_projects,
            )

    msg = f"invalid dataset {dataset}"
    raise ValueError(msg)
