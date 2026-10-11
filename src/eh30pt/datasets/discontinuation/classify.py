"""Drug discontinuation reason classification dataset."""

from collections import Counter
from functools import cache
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    NamedTuple,
    NotRequired,
    TypedDict,
    cast,
)

from datasets import Dataset
from datasets.iterable_dataset import IterableDataset
from returns.result import Failure, Success
from torch.utils.data import IterableDataset as TorchIterableDataset

from eh30pt.datasets.mixin import DatasetMixin
from eh30pt.labeling.studio import StudioClient, group_choices
from eh30pt.logging import logger
from eh30pt.settings import LabelStudioSettings
from eh30pt.types import DataSplits, FewShot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.labeling.studio import StudioTask
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


class DiscontClassifierRow(TypedDict):
    """One discontinuation reason with its event context when labeled."""

    id: int
    text: str
    phrase: NotRequired[str]
    drug: NotRequired[str]
    label: NotRequired[DiscontClassifierLabel]


class SplitRows(NamedTuple):
    """The row lists the voted rows are carved into."""

    train: list[DiscontClassifierRow]
    eval: list[DiscontClassifierRow]
    test: list[DiscontClassifierRow]


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


def labeled_row(
    *,
    task_id: int,
    text: str,
    phrase: str,
    drug: str,
    discont: Sequence[str],
    other: Sequence[str],
    meta: Sequence[str],
) -> DiscontClassifierRow | None:
    """
    Map one first-per-task annotation to a training row, or None to drop it.

    Only a lone choice trains: anything flagged by meta, asserting in both
    boxes at once, or carrying zero or several labels, is left out.
    """
    if not text or not phrase or not drug:
        return None

    if meta or (discont and other):
        return None

    choices = discont or other
    if len(choices) != 1:
        return None

    label = _map_label(choices[0])
    if label is None:
        return None

    return {
        "id": task_id,
        "text": text,
        "phrase": phrase,
        "drug": drug,
        "label": label,
    }


def labeled_rows(tasks: Sequence[StudioTask]) -> Iterator[DiscontClassifierRow]:
    """
    Yield one training row per task from its annotations' majority vote.

    Every mappable annotation votes once; ties and tasks without a mappable
    annotation train nothing. The winning row is the earliest annotation behind
    the winning label.
    """
    for task in tasks:
        data = task["data"]
        rows: list[DiscontClassifierRow] = []
        for annotation in sorted(task["annotations"], key=lambda item: item["id"]):
            grouped = group_choices(annotation, ("discont", "other", "meta"))
            row = labeled_row(
                task_id=task["id"],
                text=data.get("text", ""),
                phrase=data.get("phrase", ""),
                drug=data.get("drug", ""),
                discont=grouped["discont"],
                other=grouped["other"],
                meta=grouped["meta"],
            )
            if row is not None:
                rows.append(row)

        counts = Counter(row["label"] for row in rows)
        top = counts.most_common(2)
        if not top or (len(top) > 1 and top[0][1] == top[1][1]):
            continue

        winner = top[0][0]
        yield next(row for row in rows if row["label"] == winner)


def stream_labeled_rows(
    settings: LabelStudioSettings, labeling_projects: Sequence[int]
) -> Iterator[DiscontClassifierRow]:
    """Stream voted training rows from Label Studio projects, connecting lazily."""
    with StudioClient(settings) as studio:
        for project in labeling_projects:
            match studio.tasks(project):
                case Success(tasks):
                    yield from labeled_rows(tasks)
                case Failure(exc):
                    msg = f"could not read tasks of project {project}: {exc}"
                    raise RuntimeError(msg) from exc


def _carve(rows: Sequence[DiscontClassifierRow], *, seed: int = 0) -> SplitRows:
    """
    Carve the voted rows into train, eval and test splits.

    Rows are ordered by id first, so which split a row lands in depends on
    the rows themselves rather than the order Label Studio paged them in.
    Two tenths are held out and cut in half a second time, mirroring how
    oblique carves eval out of what its first cut held back, and rows too
    few to leave train at least one stay in train on their own.
    """
    ordered = sorted(rows, key=lambda row: row["id"])
    tenth = max(1, round(len(ordered) * 0.1))

    if len(ordered) - tenth * 2 < 1:
        return SplitRows(train=ordered, eval=[], test=[])

    dataset = Dataset.from_list(cast("list[dict[str, object]]", ordered))
    rest = dataset.train_test_split(test_size=tenth * 2, seed=seed, keep_in_memory=True)
    halves = rest["test"].train_test_split(
        test_size=tenth, seed=seed, keep_in_memory=True
    )

    return SplitRows(
        train=_rows_of(rest["train"]),
        eval=_rows_of(halves["train"]),
        test=_rows_of(halves["test"]),
    )


def _rows_of(dataset: Dataset) -> list[DiscontClassifierRow]:
    """
    Return the rows of one carved split as typed rows.

    Every row comes from `labeled_row`, which fills all four fields, so the
    arrow columns round trip back into the full typed dict.
    """
    return cast("list[DiscontClassifierRow]", dataset.to_list())


def _above(splits: SplitRows, start_after: int | None) -> SplitRows:
    """
    Drop the rows at or below `start_after` from every split.

    The filter runs after the carve, so a resumed run sees the same rows in
    the same splits it saw before, only without the consumed ones.
    """
    if start_after is None:
        return splits

    consumed: int = start_after

    def keep(rows: list[DiscontClassifierRow]) -> list[DiscontClassifierRow]:
        return [row for row in rows if row["id"] > consumed]

    return SplitRows(
        train=keep(splits.train), eval=keep(splits.eval), test=keep(splits.test)
    )


def load_discont_classifier(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
    n_virtual: int = 0,
    *,
    start_after: int | None = None,
    labeling_projects: Sequence[int],
) -> DataSplits:
    """
    Load the voted Label Studio rows as train, eval and test splits.

    The projects are read once, when the first split iterates, and all three
    splits then share that single fetch.
    """

    @cache
    def voted() -> SplitRows:
        logger.debug("reading the voted rows of projects %s", labeling_projects)
        rows = list(stream_labeled_rows(LabelStudioSettings(), labeling_projects))
        return _above(_carve(rows), start_after)

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
