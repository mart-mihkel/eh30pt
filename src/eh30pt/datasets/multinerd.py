"""MultiNERD dataset loading, prompting, and tokenization."""

from textwrap import dedent
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, cast

from datasets.iterable_dataset import IterableDataset
from datasets.load import load_dataset
from datasets.utils.info_utils import VerificationMode
from torch.utils.data import IterableDataset as TorchIterableDataset

from eh30pt.datasets.mixin import DatasetMixin
from eh30pt.logging import logger
from eh30pt.types import DataSplits, FewShot

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from datasets.arrow_dataset import Dataset
    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.types import Architecture


type MultinerdTag = Literal[
    "PER",
    "ORG",
    "LOC",
    "ANIM",
    "BIO",
    "CEL",
    "DIS",
    "EVE",
    "FOOD",
    "INST",
    "MEDIA",
    "MYTH",
    "PLANT",
    "TIME",
    "VEHI",
]

type _Lang = Literal[
    "zh",
    "nl",
    "en",
    "fr",
    "de",
    "it",
    "pl",
    "pt",
    "ru",
    "es",
]


class _Batch(TypedDict):
    tokens: list[list[str]]
    ner_tags: list[list[int]]
    lang: list[_Lang]


class MultinerdRow(TypedDict):
    """One entity sample with its sentence context."""

    sentence: str
    entity: str
    label: MultinerdTag


class Multinerd(DatasetMixin, IterableDataset, TorchIterableDataset):
    """MultiNERD NER tag classification."""

    id2label: ClassVar[Mapping[int, str]] = {
        0: "PER",
        1: "ORG",
        2: "LOC",
        3: "ANIM",
        4: "BIO",
        5: "CEL",
        6: "DIS",
        7: "EVE",
        8: "FOOD",
        9: "INST",
        10: "MEDIA",
        11: "MYTH",
        12: "PLANT",
        13: "TIME",
        14: "VEHI",
    }

    label2id: ClassVar[Mapping[str, int]] = {
        "PER": 0,
        "ORG": 1,
        "LOC": 2,
        "ANIM": 3,
        "BIO": 4,
        "CEL": 5,
        "DIS": 6,
        "EVE": 7,
        "FOOD": 8,
        "INST": 9,
        "MEDIA": 10,
        "MYTH": 11,
        "PLANT": 12,
        "TIME": 13,
        "VEHI": 14,
    }

    id2label_bio: ClassVar[Mapping[int, str]] = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-ANIM",
        8: "I-ANIM",
        9: "B-BIO",
        10: "I-BIO",
        11: "B-CEL",
        12: "I-CEL",
        13: "B-DIS",
        14: "I-DIS",
        15: "B-EVE",
        16: "I-EVE",
        17: "B-FOOD",
        18: "I-FOOD",
        19: "B-INST",
        20: "I-INST",
        21: "B-MEDIA",
        22: "I-MEDIA",
        23: "B-MYTH",
        24: "I-MYTH",
        25: "B-PLANT",
        26: "I-PLANT",
        27: "B-TIME",
        28: "I-TIME",
        29: "B-VEHI",
        30: "I-VEHI",
    }

    system_prompt = dedent(f"""
        Identify the NER tag of the entity in the sentence.
        Possible tags are: {", ".join(id2label.values())}.

        Output only the tag.
    """).strip()

    encoder_system_prompt = "Identify the NER tag of the entity in the sentence."

    @staticmethod
    def _format_user_prompt(sentence: str, entity: str) -> str:
        """Format a sentence and entity as a tag classification prompt."""
        return dedent(f"""
            Sentence: {sentence}
            Entity: {entity}
            Tag:
        """).strip()

    few_shots: ClassVar[list[FewShot]] = [
        FewShot(
            user=_format_user_prompt(
                "Chris LaCivita .",
                "Chris LaCivita",
            ),
            label="PER",
        ),
        FewShot(
            user=_format_user_prompt(
                "With FC Dynamo Moscow .",
                "FC Dynamo Moscow",
            ),
            label="ORG",
        ),
        FewShot(
            user=_format_user_prompt(
                "It is found on Fiji .",
                "Fiji",
            ),
            label="LOC",
        ),
        FewShot(
            user=_format_user_prompt(
                '" Charaxes varanes " group .',
                "Charaxes varanes",
            ),
            label="ANIM",
        ),
        FewShot(
            user=_format_user_prompt(
                '" Entomophaga " is effective against pests such '
                "as the green peach aphid .",
                "Entomophaga",
            ),
            label="BIO",
        ),
        FewShot(
            user=_format_user_prompt(
                "5 - 3754 , the closest discovered neutron star to Earth .",
                "neutron star",
            ),
            label="CEL",
        ),
        FewShot(
            user=_format_user_prompt(
                "and often obesity .",
                "obesity",
            ),
            label="DIS",
        ),
        FewShot(
            user=_format_user_prompt(
                "See Japan Golf Tour .",
                "Japan Golf Tour",
            ),
            label="EVE",
        ),
        FewShot(
            user=_format_user_prompt(
                "The mushroom is not edible .",
                "edible",
            ),
            label="FOOD",
        ),
        FewShot(
            user=_format_user_prompt(
                "Dust devils had previously only been photographed "
                'by the " Pathfinder " probe .',
                "Pathfinder",
            ),
            label="INST",
        ),
        FewShot(
            user=_format_user_prompt(
                ', as well as " National Lampoon " .',
                "National Lampoon",
            ),
            label="MEDIA",
        ),
        FewShot(
            user=_format_user_prompt(
                "This form of teasing could be called "
                '" tantalizing " , after the story of Tantalus .',
                "Tantalus",
            ),
            label="MYTH",
        ),
        FewShot(
            user=_format_user_prompt(
                "Sometimes garlic is used .",
                "garlic",
            ),
            label="PLANT",
        ),
        FewShot(
            user=_format_user_prompt(
                "The 13th century was very hard for the town .",
                "13th century",
            ),
            label="TIME",
        ),
        FewShot(
            user=_format_user_prompt(
                "It is powered by a pair of Rolls-Royce Tyne turboprop engines .",
                "Rolls-Royce Tyne",
            ),
            label="VEHI",
        ),
    ]

    def __init__(
        self,
        data: Dataset,
        tokenizer: PreTrainedTokenizerFast,
        arch: Architecture,
        n_shot: int = 0,
        n_virtual: int = 0,
        *,
        sequence_classification: bool = False,
    ) -> None:
        """Store one split with its tokenization settings."""
        self._data = data
        self._tokenizer = tokenizer
        self._arch = arch
        self._n_shot = n_shot
        self._n_virtual = n_virtual
        self._sequence_classification = sequence_classification

    def __iter__(self) -> Iterator[dict[str, list[int] | bool | int]]:
        """Yield tokenized samples for the split."""
        for row in self._unpack_row():
            encoded = self.tokenize(row, self._tokenizer, self._n_shot, self._n_virtual)
            yield cast("dict[str, list[int] | bool | int]", dict(encoded))

    def samples(self) -> Iterator[FewShot]:
        """Yield one decoder prompt and label pair per entity."""
        for idx, row in enumerate(self._unpack_row()):
            yield FewShot(
                user=self._format_user_prompt(row["sentence"], row["entity"]),
                label=row["label"],
                id=idx,
            )

    def total(self) -> None:
        """Return None because a row yields one sample per entity."""
        return

    def tokenize(
        self,
        row: MultinerdRow,
        tokenizer: PreTrainedTokenizerFast,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> BatchEncoding[list[int]]:
        """Tokenize one sample with few-shot context."""
        if n_shot > len(self.few_shots):
            msg = "requested more few shot examples than defined"
            raise ValueError(msg)

        if self._arch == "encoder":
            sep = self._tokenizer.sep_token
            user_prompt = f"{row['sentence']}{sep}{row['entity']}"
        else:
            user_prompt = self._format_user_prompt(row["sentence"], row["entity"])

        few_shots = self.few_shots[:n_shot]
        label = row["label"]

        return self._encode_example(
            arch=self._arch,
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            label=label,
            few_shots=few_shots,
            n_virtual=n_virtual,
            sequence_classification=self._sequence_classification,
        )

    def _unpack_row(self) -> Iterator[MultinerdRow]:
        """Yield one sample per entity in the split."""
        for sample in self._data:
            tokens = cast("list[str]", sample["tokens"])
            raw_tag_ids = cast("list[int]", sample["ner_tags"])
            sentence = " ".join(tokens)
            entities, tag_ids = self._join_spans(tokens, raw_tag_ids)

            for entity, tag_id in zip(entities, tag_ids, strict=True):
                if tag_id == -1:
                    continue

                yield MultinerdRow(
                    sentence=sentence,
                    entity=entity,
                    label=cast("MultinerdTag", self.id2label[tag_id]),
                )

    @staticmethod
    def _join_spans(
        tokens: list[str],
        tag_ids: list[int],
    ) -> tuple[list[str], list[int]]:
        out_ids: list[int] = []
        out_tokens: list[str] = []
        for token, tag_id in zip(tokens, tag_ids, strict=True):
            tag = Multinerd.id2label_bio[tag_id]

            if tag.startswith("B-"):
                tag = cast("MultinerdTag", tag[2:])
                out_ids.append(Multinerd.label2id[tag])
                out_tokens.append(token)
            elif tag.startswith("I-"):
                out_tokens[-1] = f"{out_tokens[-1]} {token}"
            elif tag == "O":
                out_ids.append(-1)
                out_tokens.append(token)
            else:
                tag = cast("MultinerdTag", tag)
                out_ids.append(Multinerd.label2id[tag])
                out_tokens.append(token)

        return out_tokens, out_ids


def _filter_english(batch: _Batch) -> list[bool]:
    return [lang == "en" for lang in batch["lang"]]


def load_multinerd(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
    n_virtual: int = 0,
    *,
    sequence_classification: bool = False,
) -> DataSplits:
    """Load english MultiNERD splits as per-split tokenized streaming datasets."""
    logger.debug("loading raw multinerd dataset")
    data = load_dataset(
        "Babelscape/multinerd",
        verification_mode=VerificationMode.NO_CHECKS,
    )

    logger.info("using english only multinerd subset")
    train: Dataset = data["train"].filter(_filter_english, batched=True)
    eval_: Dataset = data["validation"].filter(_filter_english, batched=True)
    test: Dataset = data["test"].filter(_filter_english, batched=True)

    return DataSplits(
        Multinerd(
            train,
            tokenizer,
            arch,
            n_shot,
            n_virtual,
            sequence_classification=sequence_classification,
        ),
        Multinerd(
            eval_,
            tokenizer,
            arch,
            n_shot,
            n_virtual,
            sequence_classification=sequence_classification,
        ),
        Multinerd(
            test,
            tokenizer,
            arch,
            n_shot,
            n_virtual,
            sequence_classification=sequence_classification,
        ),
    )
