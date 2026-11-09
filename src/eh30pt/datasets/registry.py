"""Dataset registry mapping dataset names to loaders."""

from itertools import islice
from typing import TYPE_CHECKING, Any

from eh30pt.datasets.discont.binary import DiscontBinary, load_discont_binary
from eh30pt.datasets.discont.classify import (
    DiscontClassifier,
    load_discont_classifier,
)
from eh30pt.datasets.discont.extract import (
    DiscontExtract,
    load_discont_extract,
)
from eh30pt.datasets.mixin import DatasetMixin
from eh30pt.datasets.multinerd import Multinerd, load_multinerd
from eh30pt.datasets.oblique import Obl, load_obl
from eh30pt.logging import logger
from eh30pt.types import DataSplits

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from openai.types.shared_params import ResponseFormatJSONSchema
    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.types import (
        Architecture,
        DatasetName,
        DatasetTask,
        DataSource,
        FewShot,
        TokenizedSample,
    )


class _LimitedTrain(DatasetMixin[Any]):
    """Indexable first-N view over a loaded training split."""

    def __init__(self, base: DatasetMixin[Any], limit: int) -> None:
        """Store the wrapped split with its example limit."""
        self._base = base
        self._limit = limit

    def __len__(self) -> int:
        """Return the number of retained samples."""
        return min(len(self._base), self._limit)

    def __getitem__(self, index: int) -> TokenizedSample:
        """Return a retained tokenized sample."""
        if index < -len(self) or index >= len(self):
            raise IndexError(index)

        return self._base[index if index >= 0 else len(self) + index]

    def samples(self) -> Iterator[FewShot]:
        """Yield at most the first `limit` prompt and label pairs."""
        base = self._base
        samples = base.samples() if isinstance(base, DatasetMixin) else iter(())
        yield from islice(samples, self._limit)

    def total(self) -> int:
        """Return the number of samples the limit yields."""
        base = self._base
        if not isinstance(base, DatasetMixin):
            return self._limit

        total = base.total()
        return self._limit if total is None else min(total, self._limit)

    def tokenize(
        self,
        row: Any,  # noqa: ANN401
        tokenizer: PreTrainedTokenizerFast,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> BatchEncoding[list[int]]:
        """Delegate single-row tokenization to the wrapped split."""
        base = self._base
        if isinstance(base, DatasetMixin):
            return base.tokenize(row, tokenizer, n_shot, n_virtual)

        msg = "wrapped train split does not support tokenization"
        raise TypeError(msg)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Delegate split attributes such as label maps to the wrapped split."""
        try:
            base = self.__dict__["_base"]
        except KeyError:
            msg = f"train limit has no attribute '{name}'"
            raise AttributeError(msg) from None

        return getattr(base, name)


def _is_discont(dataset: DatasetName) -> bool:
    return dataset.startswith("discont-")


def _is_classifier(dataset: DatasetName) -> bool:
    return dataset.startswith(("discont-classify", "discont-binary/extract-labels"))


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

    if dataset == "discont-classify/extract-labels":
        return tuple(DiscontClassifier.id2label.values())

    if dataset == "discont-binary/extract-labels":
        return tuple(DiscontBinary.id2label.values())

    if _is_extract(dataset):
        msg = f"dataset '{dataset}' is an extraction dataset and has no labels"
        raise ValueError(msg)

    msg = f"invalid dataset {dataset}"
    raise ValueError(msg)


def get_sys_prompt(dataset: DatasetName, arch: Architecture | None = None) -> str:
    """Return the instruction for `dataset` and `arch` without loading data."""
    if dataset == "multinerd":
        return Multinerd.system_prompt_for(arch)

    if dataset == "oblique":
        return Obl.system_prompt_for(arch)

    if dataset == "discont-classify/extract-labels":
        return DiscontClassifier.system_prompt_for(arch)

    if dataset == "discont-binary/extract-labels":
        return DiscontBinary.system_prompt_for(arch)

    if _is_extract(dataset):
        return DiscontExtract.system_prompt_for(arch)

    msg = f"invalid dataset {dataset}"
    raise ValueError(msg)


def get_positive_label(
    dataset: DatasetName,
    arch: Architecture,
    *,
    sequence_classification: bool = False,
) -> str | int | None:
    """Return the positive class of a binary `dataset` in `arch`'s label space."""
    label_arch: Architecture = "encoder" if sequence_classification else arch
    if dataset == "multinerd":
        return Multinerd.positive_label_for(label_arch)

    if dataset == "oblique":
        return Obl.positive_label_for(label_arch)

    if dataset == "discont-classify/extract-labels":
        return DiscontClassifier.positive_label_for(label_arch)

    if dataset == "discont-binary/extract-labels":
        return DiscontBinary.positive_label_for(label_arch)

    if _is_extract(dataset):
        return None

    msg = f"invalid dataset {dataset}"
    raise ValueError(msg)


def get_response_schema(dataset: DatasetName) -> ResponseFormatJSONSchema:
    """Return the structured generation schema for `dataset` without loading data."""
    if dataset == "multinerd":
        return Multinerd.response_format()

    if dataset == "oblique":
        return Obl.response_format()

    if dataset == "discont-classify/extract-labels":
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
        shots = Multinerd.examples
    elif dataset == "discont-classify/extract-labels":
        shots = DiscontClassifier.examples
    elif dataset == "discont-binary/extract-labels":
        shots = DiscontBinary.examples
    elif _is_extract(dataset):
        shots = DiscontExtract.examples
    elif dataset == "oblique":
        shots = Obl.examples
    else:
        msg = f"invalid dataset {dataset}"
        raise ValueError(msg)

    if n_shot > len(shots):
        msg = f"requested {n_shot} few shot examples, only {len(shots)} defined"
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


def _load_classifier_splits(
    tokenizer: PreTrainedTokenizerFast,
    dataset: DatasetName,
    arch: Architecture,
    *,
    n_shot: int,
    n_virtual: int,
    data_path: Path | None,
    seed: int | None,
    sequence_classification: bool = False,
) -> DataSplits:
    """Load discont classifier splits from a labels file carved with `seed`."""
    if data_path is None:
        msg = f"dataset '{dataset}' needs a labels file, pass --data-path"
        raise ValueError(msg)

    split_seed = 0 if seed is None else seed
    if seed is None:
        logger.warning("no split seed, carving discont splits with seed 0")

    if dataset == "discont-classify/extract-labels":
        return load_discont_classifier(
            tokenizer,
            arch,
            n_shot,
            n_virtual,
            data_path=data_path,
            seed=split_seed,
            sequence_classification=sequence_classification,
        )

    if dataset == "discont-binary/extract-labels":
        return load_discont_binary(
            tokenizer,
            arch,
            n_shot,
            n_virtual,
            data_path=data_path,
            seed=split_seed,
            sequence_classification=sequence_classification,
        )

    msg = f"invalid dataset {dataset}"
    raise ValueError(msg)


def load_data(
    tokenizer: PreTrainedTokenizerFast,
    dataset: DatasetName,
    arch: Architecture,
    *,
    n_shot: int = 0,
    n_virtual: int = 0,
    data_path: Path | None = None,
    seed: int | None = None,
    train_size: int | None = None,
    sequence_classification: bool = False,
) -> DataSplits:
    """Load dataset splits and optionally limit the training split."""
    if train_size is not None and train_size < 1:
        msg = f"train size must be at least 1, got {train_size}"
        raise ValueError(msg)

    if not _is_classifier(dataset) and data_path is not None:
        msg = f"labels file is only read by the discont classifiers, got '{dataset}'"
        raise ValueError(msg)

    if dataset == "multinerd":
        splits = load_multinerd(
            tokenizer,
            arch,
            n_shot,
            n_virtual,
            sequence_classification=sequence_classification,
        )
    elif dataset == "oblique":
        splits = load_obl(
            tokenizer,
            arch,
            n_shot,
            n_virtual,
            sequence_classification=sequence_classification,
        )
    elif _is_extract(dataset):
        splits = load_discont_extract(
            tokenizer,
            arch,
            n_shot,
            n_virtual,
            source=_extract_source(dataset),
        )
    elif _is_classifier(dataset):
        splits = _load_classifier_splits(
            tokenizer,
            dataset,
            arch,
            n_shot=n_shot,
            n_virtual=n_virtual,
            data_path=data_path,
            seed=seed,
            sequence_classification=sequence_classification,
        )
    else:
        msg = f"invalid dataset {dataset}"
        raise ValueError(msg)

    if train_size is None:
        return splits

    logger.info("limiting train split to %d examples", train_size)
    return DataSplits(
        _LimitedTrain(splits.train, train_size),
        splits.eval,
        splits.test,
    )
