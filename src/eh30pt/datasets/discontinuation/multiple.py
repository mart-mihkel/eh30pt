"""Drug discontinuation multiple-event classification dataset."""

from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, cast

from datasets.iterable_dataset import IterableDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from eh30pt.datasets.discontinuation.streaming import _stream_rows
from eh30pt.datasets.mixin import DatasetMixin
from eh30pt.logging import logger
from eh30pt.settings import PostgresSettings
from eh30pt.types import DatasetTask, DataSplits, FewShot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.types import Architecture, DataSource


type DiscontMultipleLabel = Literal["none", "single", "multiple"]


class DiscontMultipleRow(TypedDict):
    """One candidate discontinuation event."""

    id: int
    text: str


class DiscontMultiple(DatasetMixin, IterableDataset, TorchIterableDataset):
    """Drug discontinuation multiple-event classification."""

    task: ClassVar[DatasetTask] = "classification"

    id2label: ClassVar[Mapping[int, DiscontMultipleLabel]] = {
        0: "none",
        1: "single",
        2: "multiple",
    }

    label2id: ClassVar[Mapping[DiscontMultipleLabel, int]] = {
        "none": 0,
        "single": 1,
        "multiple": 2,
    }

    system_prompt = dedent(
        """
        You are a helpful assistant that reads Estonian electronic health records written by doctors about drug discontinuation for their patients on medication.
        For each input, determine whether the event describes no discontinuation, a single discontinuation, or multiple discontinuations.
        Your input is the extracted reason for stopping the drug.

        Here are the categories you must use:

        1. none: The text contains no discontinuation event.
        2. single: The text describes a single discontinuation event (one drug stopped).
        3. multiple: The text describes multiple discontinuation events (multiple drugs stopped).

        Only choose one category per input. Respond only with the category name.
        """  # noqa: E501
    ).strip()

    few_shots: ClassVar[list[FewShot]] = [
        FewShot(user="tahab xanaxit", label="none"),
        FewShot(user="katkestas nebileti tarvitamise, vererõhk normal", label="single"),
        FewShot(user="lõpetas ibuprofeeni, aspiriini tarvitamise", label="multiple"),
    ]

    def __init__(
        self,
        rows: Callable[[], Iterator[DiscontMultipleRow]],
        tokenizer: PreTrainedTokenizerFast,
        arch: Architecture,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> None:
        """Store the row factory with its tokenization settings."""
        self._rows = rows
        self._tokenizer = tokenizer
        self._arch = arch
        self._n_shot = n_shot
        self._n_virtual = n_virtual
        self._warned = False

    def __iter__(self) -> Iterator[dict[str, list[int] | bool | int]]:
        """Yield tokenized rows for the split."""
        for row in self._rows():
            encoded = self.tokenize(row, self._tokenizer, self._n_shot, self._n_virtual)
            yield cast("dict[str, list[int] | bool | int]", dict(encoded))

    def samples(self) -> Iterator[FewShot]:
        """Yield decoder prompt and label pairs for the split."""
        if not self._warned:
            logger.warning("discontinuation multiple dataset has no labels")
            self._warned = True

        for row in self._rows():
            yield FewShot(user=row["text"], label="", id=row["id"])

    def tokenize(
        self,
        row: DiscontMultipleRow,
        tokenizer: PreTrainedTokenizerFast,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> BatchEncoding[list[int]]:
        """Tokenize one row with few-shot context."""
        if not self._warned:
            logger.warning("discontinuation multiple dataset has no labels")
            self._warned = True

        if n_shot > len(self.few_shots):
            msg = "requested more few shot examples than defined"
            raise ValueError(msg)

        user_prompt = row["text"]
        few_shots = self.few_shots[:n_shot]
        label = ""

        return self._encode_example(
            arch=self._arch,
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            label=label,
            few_shots=few_shots,
            n_virtual=n_virtual,
        )


def _no_multiple_rows() -> Iterator[DiscontMultipleRow]:
    """Yield no rows for the empty train and eval splits."""
    return iter(())


def _stream_multiple_rows(
    url: str,
    source: DataSource,
    *,
    page_size: int,
    start_after: int | None = None,
) -> Iterator[DiscontMultipleRow]:
    """Stream discontinuation rows with id above start_after, connecting lazily."""
    yield from cast(
        "Iterator[DiscontMultipleRow]",
        _stream_rows(
            url,
            source,
            page_size=page_size,
            start_after=start_after,
            message="streaming multiple discontinuation events",
        ),
    )


def load_discont_multiple(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
    n_virtual: int = 0,
    *,
    page_size: int = 10000,
    source: DataSource = "trigger",
    start_after: int | None = None,
) -> DataSplits:
    """Load multiple-discontinuation candidates as splits with only test populated."""
    logger.debug(
        "multiple discontinuation rows stream from Est-Health on first iteration"
    )

    url = PostgresSettings().connection_url
    test_rows = partial(
        _stream_multiple_rows,
        url,
        page_size=page_size,
        source=source,
        start_after=start_after,
    )

    return DataSplits(
        train=DiscontMultiple(_no_multiple_rows, tokenizer, arch, n_shot, n_virtual),
        eval=DiscontMultiple(_no_multiple_rows, tokenizer, arch, n_shot, n_virtual),
        test=DiscontMultiple(test_rows, tokenizer, arch, n_shot, n_virtual),
    )
