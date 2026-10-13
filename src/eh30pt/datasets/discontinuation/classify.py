"""Drug discontinuation reason classification dataset."""

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


type DiscontClassifierLabel = Literal[
    "Adverse reactions",
    "Treatment success",
    "Treatment inefficacy",
    "Contraindication",
    "Other",
    "Adverse reaction without discontinuation",
    "Dosage change",
    "Other non-discontinuation",
]

DiscontClassifierRow = LabeledRow[DiscontClassifierLabel]
"""One discontinuation event carrying one of the eight reason labels."""


class DiscontClassifier(DatasetMixin, IterableDataset, TorchIterableDataset):
    """Drug discontinuation reason classification."""

    id2label: ClassVar[Mapping[int, DiscontClassifierLabel]] = {
        0: "Adverse reactions",
        1: "Treatment success",
        2: "Treatment inefficacy",
        3: "Contraindication",
        4: "Other",
        5: "Adverse reaction without discontinuation",
        6: "Dosage change",
        7: "Other non-discontinuation",
    }

    label2id: ClassVar[Mapping[DiscontClassifierLabel, int]] = {
        "Adverse reactions": 0,
        "Treatment success": 1,
        "Treatment inefficacy": 2,
        "Contraindication": 3,
        "Other": 4,
        "Adverse reaction without discontinuation": 5,
        "Dosage change": 6,
        "Other non-discontinuation": 7,
    }

    system_prompt = dedent(
        """
        You are a helpful assistant that reads Estonian electronic health records written by doctors about drug discontinuation for their patients on medication. For each input, classify the reason why the patient stopped taking the medication.

        Your input is the clinical text together with the extracted phrase describing the event and the drug in question.

        Here are the categories you must use:

        1. Adverse reactions: Discontinuation due to adverse side effects, allergic reactions, or negative interactions with other medications.
        2. Treatment success: Discontinuation due to successful treatment completion or sufficient improvement in health.
        3. Treatment inefficacy: Perceived ineffectiveness of the treatment or loss of belief in its efficacy.
        4. Contraindication: Discontinuation due to the emergence or discovery of a medical condition or risk factor that makes the continued use of the treatment unsafe or inappropriate (e.g. "vastunäidustatud").
        5. Other: Other medical reasons not explicitly covered by the other categories.
        6. Adverse reaction without discontinuation: Adverse side effects, allergic reactions, or negative interactions where treatment continued.
        7. Dosage change: The drug dose was increased or decreased rather than stopped.
        8. Other non-discontinuation: An event that is neither a discontinuation nor any of the reasons above.

        Only choose one category per input. If the reason is ambiguous or fits none of the categories above, select "Other". Respond only with the category name.
        """  # noqa: E501
    ).strip()

    few_shots: ClassVar[list[FewShot]] = [
        FewShot(user="ei talu", label="Adverse reactions"),
        FewShot(user="vererõhk püsivalt normis", label="Treatment success"),
        FewShot(user="valu ei leevenenud, ebaefektiivne", label="Treatment inefficacy"),
        FewShot(user="soovib rasestuda, katkestab ravi", label="Contraindication"),
        FewShot(user="patsient viidi üle insuliinravile", label="Other"),
        FewShot(
            user="iiveldus tekkis, aga ravi jätkub",
            label="Adverse reaction without discontinuation",
        ),
        FewShot(
            user="vähendasime statiini doosi",
            label="Dosage change",
        ),
        FewShot(
            user="patsient jätkab ravi muutuseta",
            label="Other non-discontinuation",
        ),
    ]

    @staticmethod
    def _format_user_prompt(text: str, phrase: str, drug: str) -> str:
        """Format clinical text with its phrase and drug as a prompt."""
        return dedent(f"""
            Text: {text}
            Phrase: {phrase}
            Drug: {drug}
            Category:
        """).strip()

    def __init__(
        self,
        rows: Callable[[], Iterator[DiscontClassifierRow]],
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
        row: DiscontClassifierRow,
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
    def _format_user(cls, row: DiscontClassifierRow) -> str:
        """
        Format a row, falling back to bare text without extracted fields.

        Event rows streamed for evaluation predate phrase and drug extraction,
        so only labeled rows carry them.
        """
        phrase = row.get("phrase", "")
        drug = row.get("drug", "")
        if not phrase or not drug:
            return row["text"]

        return cls._format_user_prompt(row["text"], phrase, drug)


def _map_label(choice: str) -> DiscontClassifierLabel | None:
    """
    Return the choice when it names a classifier label, or None otherwise.

    The discont box and the classifier share one label set, so mapping is a
    membership check rather than a translation table.
    """
    if choice not in DiscontClassifier.id2label.values():
        return None

    return cast("DiscontClassifierLabel", choice)


def load_discont_classifier(
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
    def voted() -> SplitRows[DiscontClassifierLabel]:
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
        train=DiscontClassifier(
            lambda: iter(voted().train), tokenizer, arch, n_shot, n_virtual
        ),
        eval=DiscontClassifier(
            lambda: iter(voted().eval), tokenizer, arch, n_shot, n_virtual
        ),
        test=DiscontClassifier(
            lambda: iter(voted().test), tokenizer, arch, n_shot, n_virtual
        ),
    )
