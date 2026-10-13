"""Binary drug discontinuation classification dataset."""

from functools import cache
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    cast,
)

from datasets.iterable_dataset import IterableDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from eh30pt.datasets.mixin import DatasetMixin
from eh30pt.labeling.votes import (
    LabeledRow,
    SplitRows,
    above,
    carve,
    stream_labeled_rows,
)
from eh30pt.logging import logger
from eh30pt.settings import LabelStudioSettings
from eh30pt.types import DataSplits, FewShot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.types import Architecture


type DiscontBinaryLabel = Literal["yes", "no"]


DiscontBinaryRow = LabeledRow[DiscontBinaryLabel]
"""One discontinuation event answered yes or no."""


class DiscontBinary(DatasetMixin, IterableDataset, TorchIterableDataset):
    """Whether an event discontinued the medication, whatever the reason."""

    id2label: ClassVar[Mapping[int, DiscontBinaryLabel]] = {
        0: "yes",
        1: "no",
    }

    label2id: ClassVar[Mapping[DiscontBinaryLabel, int]] = {
        "yes": 0,
        "no": 1,
    }

    reason2label: ClassVar[Mapping[str, DiscontBinaryLabel]] = {
        "Adverse reactions": "yes",
        "Treatment success": "yes",
        "Treatment inefficacy": "yes",
        "Contraindication": "yes",
        "Other": "yes",
        "Adverse reaction without discontinuation": "no",
        "Dosage change": "no",
        "Other non-discontinuation": "no",
    }
    """
    How each reason label folds down to the binary question.
    """

    system_prompt = dedent(
        """
        You are a helpful assistant that reads Estonian electronic health records written by doctors about drug discontinuation for their patients on medication. For each input, decide whether the event discontinued the medication or not.
        Your input is the clinical text together with the extracted phrase describing the event and the drug in question.

        Answer "yes" when the patient stopped taking the medication, whatever the reason: adverse reactions, completed treatment, perceived ineffectiveness, a contraindication, or another medical or non-medical reason.
        Answer "no" otherwise.

        Respond only with "yes" or "no".
        """  # noqa: E501
    ).strip()

    few_shots: ClassVar[list[FewShot]] = [
        FewShot(user="ei talu", label="yes"),
        FewShot(user="vererõhk püsivalt normis", label="yes"),
        FewShot(user="valu ei leevenenud, ebaefektiivne", label="yes"),
        FewShot(user="soovib rasestuda, katkestab ravi", label="yes"),
        FewShot(user="patsient viidi üle insuliinravile", label="yes"),
        FewShot(user="iiveldus tekkis, aga ravi jätkub", label="no"),
        FewShot(user="vähendasime statiini doosi", label="no"),
        FewShot(user="patsient jätkab ravi muutuseta", label="no"),
    ]

    @staticmethod
    def _format_user_prompt(text: str, phrase: str, drug: str) -> str:
        """Format clinical text with its phrase and drug as a prompt."""
        return dedent(f"""
            Text: {text}
            Phrase: {phrase}
            Drug: {drug}
            Discontinuation:
        """).strip()

    def __init__(
        self,
        rows: Callable[[], Iterator[DiscontBinaryRow]],
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

    def __iter__(self) -> Iterator[dict[str, list[int] | bool | int]]:
        """Yield tokenized rows for the split."""
        for row in self._rows():
            encoded = self.tokenize(row, self._tokenizer, self._n_shot, self._n_virtual)
            yield cast("dict[str, list[int] | bool | int]", dict(encoded))

    def samples(self) -> Iterator[FewShot]:
        """Yield prompt and label pairs for the split."""
        for row in self._rows():
            yield FewShot(
                user=self._format_user(row),
                label=row.get("label", ""),
                id=row["id"],
            )

    def tokenize(
        self,
        row: DiscontBinaryRow,
        tokenizer: PreTrainedTokenizerFast,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> BatchEncoding[list[int]]:
        """Tokenize one row with few-shot context."""
        if n_shot > len(self.few_shots):
            msg = "requested more few shot examples than defined"
            raise ValueError(msg)

        few_shots = self.few_shots[:n_shot]

        return self._encode_example(
            arch=self._arch,
            tokenizer=tokenizer,
            user_prompt=self._format_user(row),
            label=row.get("label", ""),
            few_shots=few_shots,
            n_virtual=n_virtual,
        )

    @classmethod
    def _format_user(cls, row: DiscontBinaryRow) -> str:
        """Format a row, falling back to bare text without extracted fields."""
        phrase = row.get("phrase", "")
        drug = row.get("drug", "")
        if not phrase or not drug:
            return row["text"]

        return cls._format_user_prompt(row["text"], phrase, drug)


def _map_label(choice: str) -> DiscontBinaryLabel | None:
    return DiscontBinary.reason2label.get(choice)


def load_discont_binary(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
    n_virtual: int = 0,
    *,
    start_after: int | None = None,
    seed: int,
    labeling_projects: Sequence[int],
) -> DataSplits:
    """
    Load the voted Label Studio rows as train, eval and test splits.

    The projects are read once, when the first split iterates, and all three
    splits then share that single fetch.
    """

    @cache
    def voted() -> SplitRows[DiscontBinaryLabel]:
        logger.debug("reading the voted rows of projects %s", labeling_projects)
        rows = list(
            stream_labeled_rows(
                LabelStudioSettings(),
                labeling_projects,
                _map_label,
            )
        )
        splits = above(carve(rows, seed=seed), start_after)
        logger.debug(
            "carved %d voted rows with seed %d into train=%d eval=%d test=%d",
            len(rows),
            seed,
            len(splits.train),
            len(splits.eval),
            len(splits.test),
        )

        return splits

    return DataSplits(
        train=DiscontBinary(
            lambda: iter(voted().train), tokenizer, arch, n_shot, n_virtual
        ),
        eval=DiscontBinary(
            lambda: iter(voted().eval), tokenizer, arch, n_shot, n_virtual
        ),
        test=DiscontBinary(
            lambda: iter(voted().test), tokenizer, arch, n_shot, n_virtual
        ),
    )
