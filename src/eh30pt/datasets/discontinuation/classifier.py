"""Drug discontinuation reason classification dataset."""

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


type DiscontClassifierLabel = Literal[
    "Adverse reactions",
    "Treatment success",
    "Treatment inefficacy",
    "Contraindication",
    "Non-medical reasons",
    "Other",
    "Indeterminate",
]


class DiscontClassifierRow(TypedDict):
    """One labeled discontinuation reason."""

    id: int
    text: str


class DiscontClassifier(DatasetMixin, IterableDataset, TorchIterableDataset):
    """Drug discontinuation reason classification."""

    id2label: ClassVar[Mapping[int, DiscontClassifierLabel]] = {
        0: "Adverse reactions",
        1: "Treatment success",
        2: "Treatment inefficacy",
        3: "Contraindication",
        4: "Non-medical reasons",
        5: "Other",
        6: "Indeterminate",
    }

    label2id: ClassVar[Mapping[DiscontClassifierLabel, int]] = {
        "Adverse reactions": 0,
        "Treatment success": 1,
        "Treatment inefficacy": 2,
        "Contraindication": 3,
        "Non-medical reasons": 4,
        "Other": 5,
        "Indeterminate": 6,
    }

    system_prompt = dedent(
        """
        You are a helpful assistant that reads Estonian electronic health records written by doctors about drug discontinuation for their patients on medication. For each input, classify the reason why the patient stopped taking the medication.

        Your input is the extracted reason for stopping the drug.

        Here are the categories you must use:

        1. Adverse reactions: Discontinuation due to adverse side effects, allergic reactions, or negative interactions with other medications.
        2. Treatment success: Discontinuation due to successful treatment completion or sufficient improvement in health.
        3. Treatment inefficacy: Perceived ineffectiveness of the treatment or loss of belief in its efficacy.
        4. Contraindication: Discontinuation due to the emergence or discovery of a medical condition or risk factor that makes the continued use of the treatment unsafe or inappropriate (e.g. "vastunäidustatud").
        5. Non-medical reasons: Discontinuation due to factor unrelated to the patient's health or the treatment's medical effects, such as financial constraints, access issues, personal choice, cultural beliefs, or social circumstances.
        6. Other: Other medical reasons not explicitly covered by the other categories.
        7. Indeterminate: Unclear or unspecified reasons for discontinuation.

        Only choose one category per input. If the reason is ambiguous or not clearly stated, select "Indeterminate". Respond only with the category name.
        """  # noqa: E501
    ).strip()

    few_shots: ClassVar[list[FewShot]] = [
        FewShot(user="tekkis tugev iiveldus ja peapööritus", label="Adverse reactions"),
        FewShot(user="vererõhk püsivalt normis", label="Treatment success"),
        FewShot(user="valu ei leevenenud, ebaefektiivne", label="Treatment inefficacy"),
        FewShot(user="raseduse tõttu vastunäidustatud", label="Contraindication"),
        FewShot(user="jättis rohu ostmata, raha pole", label="Non-medical reasons"),
        FewShot(user="ravim ära jäetud", label="Indeterminate"),
        FewShot(user="patsient viidi üle insuliinravile", label="Other"),
    ]

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
        self._warned = False

    def __iter__(self) -> Iterator[dict[str, list[int] | bool | int]]:
        """Yield tokenized rows for the split."""
        for row in self._rows():
            encoded = self.tokenize(row, self._tokenizer, self._n_shot, self._n_virtual)
            yield cast("dict[str, list[int] | bool | int]", dict(encoded))

    def samples(self) -> Iterator[FewShot]:
        """Yield prompt and label pairs for the split."""
        if not self._warned:
            logger.warning("discontinuation classifier dataset has no labels")
            self._warned = True

        for row in self._rows():
            yield FewShot(user=row["text"], label="", id=row["id"])

    def tokenize(
        self,
        row: DiscontClassifierRow,
        tokenizer: PreTrainedTokenizerFast,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> BatchEncoding[list[int]]:
        """Tokenize one row with few-shot context."""
        if not self._warned:
            logger.warning("discontinuation classifier dataset has no labels")
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


def _no_cls_rows() -> Iterator[DiscontClassifierRow]:
    """Yield no rows for the empty train and eval splits."""
    return iter(())


def _stream_classifier_rows(
    url: str,
    source: DataSource,
    *,
    page_size: int,
    start_after: int | None = None,
) -> Iterator[DiscontClassifierRow]:
    """Stream discontinuation rows with id above start_after, connecting lazily."""
    yield from cast(
        "Iterator[DiscontClassifierRow]",
        _stream_rows(
            url,
            source,
            page_size=page_size,
            start_after=start_after,
            message="streaming discontinuation events",
        ),
    )


def load_discont_classifier(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
    n_virtual: int = 0,
    *,
    page_size: int = 10000,
    source: DataSource = "trigger",
    start_after: int | None = None,
) -> DataSplits:
    """Load discontinuation candidates as splits with only test populated."""
    logger.debug("discontinuation rows stream from Est-Health on first iteration")
    url = PostgresSettings().connection_url
    test_rows = partial(
        _stream_classifier_rows,
        url,
        page_size=page_size,
        source=source,
        start_after=start_after,
    )

    return DataSplits(
        train=DiscontClassifier(_no_cls_rows, tokenizer, arch, n_shot, n_virtual),
        eval=DiscontClassifier(_no_cls_rows, tokenizer, arch, n_shot, n_virtual),
        test=DiscontClassifier(test_rows, tokenizer, arch, n_shot, n_virtual),
    )
