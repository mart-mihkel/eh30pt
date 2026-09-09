"""Dataset registry: map dataset names to pipeline classes."""

from typing import TYPE_CHECKING

from eh30pt.datasets.multinerd import Multinerd
from eh30pt.datasets.oblique import Oblique

if TYPE_CHECKING:
    from datasets.splits import Split
    from transformers import PreTrainedTokenizerFast

    from eh30pt.types import Architecture, DatasetName, DatasetProtocol


def get_dataset(
    dataset: DatasetName,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
) -> DatasetProtocol:
    """Instantiate the pipeline class for `dataset` without loading data."""
    if dataset == "multinerd":
        return Multinerd(tokenizer, arch)

    return Oblique(tokenizer, arch)


def get_sys_prompt(
    dataset: DatasetName,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
) -> str:
    """Return the system prompt for `dataset` without loading data."""
    return get_dataset(dataset, tokenizer, arch).system_prompt


def load_data(
    tokenizer: PreTrainedTokenizerFast,
    dataset: DatasetName,
    arch: Architecture,
    *,
    n_shot: int = 0,
    n_virtual: int = 0,
    split: Split | None = None,
) -> DatasetProtocol:
    """Load and prompt-format `dataset`, returning the pipeline."""
    ds = get_dataset(dataset, tokenizer, arch)
    ds.load(n_shot=n_shot, n_virtual=n_virtual, split=split)

    return ds
