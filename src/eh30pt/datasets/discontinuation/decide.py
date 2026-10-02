"""Drug discontinuation decision attribution dataset."""

from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, cast

from datasets.iterable_dataset import IterableDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from eh30pt.datasets.discontinuation.streaming import _stream_rows
from eh30pt.datasets.mixin import DatasetMixin
from eh30pt.logging import logger
from eh30pt.settings import PostgresSettings
from eh30pt.types import DataSplits, FewShot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.types import Architecture, DataSource


type DeciderLabel = Literal[
    "Doctor",
    "Patient",
    "Unspecified",
]


class DeciderRow(TypedDict):
    """One candidate discontinuation text."""

    id: int
    text: str


class DiscontDecider(DatasetMixin, IterableDataset, TorchIterableDataset):
    """Drug discontinuation decision attribution."""

    id2label: ClassVar[Mapping[int, DeciderLabel]] = {
        0: "Doctor",
        1: "Patient",
        2: "Unspecified",
    }

    label2id: ClassVar[Mapping[DeciderLabel, int]] = {
        "Doctor": 0,
        "Patient": 1,
        "Unspecified": 2,
    }

    system_prompt = dedent(
        """
        You are a helpful assistant that reads Estonian electronic health records written by doctors about drug discontinuation for their patients on medication. For each input, determine who made the decision to stop the use of a medication: the patient or the doctor.
        Your input is the extracted phrase about stopping the drug and the full anamnesis for context.

        Here are the categories you must use:

        1. Doctor: The doctor made the decision that the patient should stop the medications.
        2. Patient: The patient stopped taking the medications without a clear signal from the doctor.
        3. Unspecified: Based on the text, it is unclear who made the decision.

        Only choose one category per input. Respond only with the category name.
        """  # noqa: E501
    ).strip()

    few_shots: ClassVar[list[FewShot]] = [
        FewShot(user="katkestas, iiveldus ja peavalu", label="Patient"),
        FewShot(user="katkestas losartaani, operatsioonieelse pausi", label="Doctor"),
        FewShot(user="kolesterool olnud normis", label="Unspecified"),
        FewShot(user="patsient jättis rohu võtmata", label="Patient"),
        FewShot(user="peatatud kõrgenenud maksaproovide tõttu", label="Doctor"),
    ]

    def __init__(
        self,
        rows: Callable[[], Iterator[DeciderRow]],
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
        """Yield prompt and label pairs for the split."""
        if not self._warned:
            logger.warning("discontinuation decider dataset has no labels")
            self._warned = True

        for row in self._rows():
            yield FewShot(user=row["text"], label="", id=row["id"])

    def tokenize(
        self,
        row: DeciderRow,
        tokenizer: PreTrainedTokenizerFast,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> BatchEncoding[list[int]]:
        """Tokenize one row with few-shot context."""
        if not self._warned:
            logger.warning("discontinuation decider dataset has no labels")
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


def _no_decider_rows() -> Iterator[DeciderRow]:
    """Yield no rows for the empty train and eval splits."""
    return iter(())


def _stream_decider_rows(
    url: str,
    source: DataSource,
    *,
    page_size: int,
    start_after: int | None = None,
) -> Iterator[DeciderRow]:
    """Stream decider rows with id above start_after, connecting lazily."""
    yield from cast(
        "Iterator[DeciderRow]",
        _stream_rows(
            url,
            source,
            page_size=page_size,
            start_after=start_after,
            message="streaming high-recall discontinuation events",
        ),
    )


def load_discont_decider(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
    n_virtual: int = 0,
    *,
    page_size: int = 10000,
    source: DataSource = "trigger",
    start_after: int | None = None,
) -> DataSplits:
    """Load decider candidates as splits with only test populated."""
    logger.debug("decider rows stream from Est-Health on first iteration")
    url = PostgresSettings().connection_url
    test_rows = partial(
        _stream_decider_rows,
        url,
        page_size=page_size,
        source=source,
        start_after=start_after,
    )

    return DataSplits(
        train=DiscontDecider(_no_decider_rows, tokenizer, arch, n_shot, n_virtual),
        eval=DiscontDecider(_no_decider_rows, tokenizer, arch, n_shot, n_virtual),
        test=DiscontDecider(test_rows, tokenizer, arch, n_shot, n_virtual),
    )
