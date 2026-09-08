"""OBL (oblique phrase removal) dataset loading, prompting, and tokenization."""

from textwrap import dedent
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, TypedDict, cast

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict

from eh30pt.constants import SENTINEL_TOKEN
from eh30pt.datasets.truncation import get_max_length
from eh30pt.logging import logger
from eh30pt.types import DatasetProtocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from datasets.splits import Split
    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.types import Architecture


type _Label = Literal["seotud", "vaba", "ebaloomulik", "liigne koma", "kaheldav"]


class _Sample(TypedDict):
    id: int
    fpath: str
    sentence: str
    remove_start: int
    remove_end: int
    removed: str
    label: _Label
    short_sent: str
    cons_score: float
    ual: float
    la: float


class _Example(NamedTuple):
    user: str
    assistant: str


class Oblique(DatasetProtocol):
    """
    Estonian oblique-phrase removal classification with architecture prompts.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerFast
        Tokenizer used for prompting and tokenization.
    arch : Architecture
        Model architecture, selecting the prompt format.

    """

    id2label: ClassVar[Mapping[int, _Label]] = {
        0: "seotud",
        1: "ebaloomulik",
        2: "liigne koma",
        3: "vaba",
        4: "kaheldav",
    }

    label2id: ClassVar[Mapping[_Label, int]] = {
        "seotud": 0,
        "ebaloomulik": 1,
        "liigne koma": 2,
        "vaba": 3,
        "kaheldav": 4,
    }

    _en_to_et: ClassVar[Mapping[str, _Label]] = {
        "bound": "seotud",
        "free": "vaba",
        "unnatural": "ebaloomulik",
        "redundant comma": "liigne koma",
        "dubious": "kaheldav",
    }

    _permalink: ClassVar[str] = (
        "https://raw.githubusercontent.com/estnltk/estnltk-model-data/"
        "c6f373a49d29a5a4b5e92326e5309b173bfcd3d1"
        "/phrase_removal/benchmarks/obl_phrases/obl_max/data/obl_all_hand_annotated.csv"
    )

    def __init__(self, tokenizer: PreTrainedTokenizerFast, arch: Architecture) -> None:
        """Store the tokenizer and architecture."""
        self._tokenizer = tokenizer
        self._arch = arch
        self._data: DatasetDict | None = None
        self._raw_data: DatasetDict | None = None

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the configured architecture."""
        if self._arch == "encoder":
            return (
                "Mis on seos fragmendi ja lause vahel,\n"
                "kas see on seotud, vaba, kaheldav, ebaloomulik "
                f"või liigne koma?{self._tokenizer.sep_token}"
            )

        if self._arch == "decoder":
            return dedent(
                """
                Sa oled liigse fraasi märgendamise mudel.
                Liigita eemaldatud fraas ühte kategooriasse:

                - liigne koma: ebavajalik või vale koma
                - ebaloomulik: grammatiliselt korrektne, kuid kohmakas või mittekeeleomane
                - kaheldav: ebaselge või piiripealne juhtum
                - seotud: vajalik grammatilise struktuuri jaoks, eemaldamine rikub grammatika
                - vaba: eemaldatav ilma tähendust või grammatikat muutmata

                Väljasta ainult kategooria.
            """  # noqa: E501
            ).strip()

        return dedent(
            """
            liigse fraasi märgendamine: liigita eemaldatud tekst ühte viiest kategooriast.

            liigne koma: ebavajalik või vale koma
            ebaloomulik: grammatiliselt valikuline, kuid kohmakas või mittekeeleomane
            kaheldav: ebaselge või piiripealne juhtum
            seotud: vajalik grammatilise struktuuri jaoks; eemaldamine rikub grammatika
            vaba: eemaldatav ilma tähendust või grammatikat muutmata

            väljasta ainult kategooria
        """  # noqa: E501
        ).strip()

    @property
    def data(self) -> DatasetDict:
        """Return the loaded dataset splits."""
        if self._data is None:
            msg = "dataset not loaded, call load() first"
            raise RuntimeError(msg)

        return self._data

    @property
    def raw_data(self) -> DatasetDict:
        """Return the untokenized dataset splits."""
        if self._raw_data is None:
            msg = "dataset not loaded, call load() first"
            raise RuntimeError(msg)

        return self._raw_data

    def load(
        self,
        *,
        n_shot: int = 0,
        n_virtual: int = 0,
        split: Split | None = None,
    ) -> None:
        """
        Load, tokenize, and prompt-format the OBL dataset into instance state.

        Parameters
        ----------
        n_shot : int
            Number of few-shot examples prepended to each prompt.
        n_virtual : int
            Number of virtual tokens reserved in the truncation budget.
        split : Split | None
            Ignored, the full dataset is always loaded.

        """
        del split

        self._raw_data = self._load_raw()
        self._data = self._tokenize(n_shot, n_virtual)

    def _load_raw(self) -> DatasetDict:
        logger.debug("loading obl csv from github permalink")
        raw = Dataset.from_csv(self._permalink, sep=";")

        logger.debug("rename 'type' to 'label'")
        raw = raw.rename_column("type", "label")

        logger.debug("translate labels")
        raw = raw.map(self._translate_en_to_et)

        s1 = raw.train_test_split(test_size=1000, seed=0)
        s2 = s1["train"].train_test_split(test_size=128, seed=0)

        return DatasetDict(
            {
                "train": s2["train"],
                "validation": s2["test"],
                "test": s1["test"],
            }
        )

    def _tokenize(self, n_shot: int, n_virtual: int) -> DatasetDict:
        logger.debug("tokenizing oblique")

        examples = self._sample_examples(n_shot)
        data = self.raw_data.map(
            self._tokenize_example,
            fn_kwargs={"examples": examples, "n_virtual": n_virtual},
            remove_columns=[
                "id",
                "fpath",
                "sentence",
                "remove_start",
                "remove_end",
                "removed",
                "label",
                "short_sent",
                "cons_score",
                "ual",
                "la",
            ],
        )

        for subsplit in data:
            logger.debug("tokenized %d %s samples", len(data[subsplit]), subsplit)

        return data

    def _build_user_prompt(self, sample: _Sample) -> str:
        if self._arch == "encoder":
            return f"{sample['removed']}{self._tokenizer.sep_token}{sample['sentence']}"

        if self._arch == "decoder":
            return dedent(f"""
                Lause: {sample["sentence"]}
                Fraas: {sample["removed"]}
                Kategooria:
            """).strip()

        return dedent(f"""
            lause: {sample["sentence"]}
            fraas: {sample["removed"]}
            kategooria:
        """).strip()

    def _format_examples_legacy(self, examples: list[_Example]) -> str:
        if not examples:
            return ""

        return "\n".join(
            f"{example.user} {example.assistant}\n" for example in examples
        )

    def _format_examples_template(
        self,
        examples: list[_Example],
    ) -> list[dict[str, str]]:
        if self._tokenizer.chat_template is None:
            msg = "attempt to format with no chat template, use _format_examples_legacy"
            raise RuntimeError(msg)

        conversation: list[dict[str, str]] = []
        for example in examples:
            conversation.append({"role": "user", "content": example.user})
            conversation.append({"role": "assistant", "content": example.assistant})

        return conversation

    def _encode_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        max_length: int | None,
        examples: list[_Example],
    ) -> BatchEncoding[list[int]]:
        if self._tokenizer.chat_template is None:
            examples_prompt = self._format_examples_legacy(examples)
            user_prompt = f"{examples_prompt}\n{user_prompt}"
            encoded_prompt = self._tokenizer(
                f"{system_prompt}\n{user_prompt}",
                truncation=True,
                max_length=max_length,
            )
        else:
            conversation: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt}
            ]

            conversation.extend(self._format_examples_template(examples))
            conversation.append({"role": "user", "content": user_prompt})

            encoded_prompt = self._tokenizer.apply_chat_template(
                conversation,
                truncation=True,
                return_dict=True,
                max_length=max_length,
                add_generation_prompt=self._arch != "encoder",
            )

        return cast("BatchEncoding[list[int]]", encoded_prompt)

    def _encode_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        label: str,
        max_length: int | None,
        examples: list[_Example],
    ) -> BatchEncoding[list[int]]:
        """Encode the prompt together with the gold label."""
        if self._tokenizer.chat_template is None:
            examples_prompt = self._format_examples_legacy(examples)
            user_prompt = f"{examples_prompt}\n{user_prompt}"
            encoded_answer = self._tokenizer(
                f"{system_prompt}\n{user_prompt} {label}",
                truncation=True,
                max_length=max_length,
            )
        else:
            conversation: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt}
            ]

            conversation.extend(self._format_examples_template(examples))
            conversation.append({"role": "user", "content": user_prompt})
            conversation.append({"role": "assistant", "content": label})

            encoded_answer = self._tokenizer.apply_chat_template(
                conversation,
                truncation=True,
                return_dict=True,
                max_length=max_length,
            )

        return cast("BatchEncoding[list[int]]", encoded_answer)

    def _tokenize_example(
        self,
        sample: _Sample,
        examples: list[_Example],
        n_virtual: int = 0,
    ) -> BatchEncoding:
        max_length = get_max_length(self._tokenizer, n_virtual)
        label = sample["label"]

        user_prompt = self._build_user_prompt(sample)
        system_prompt = self.system_prompt
        encoded_prompt = self._encode_prompt(
            system_prompt,
            user_prompt,
            max_length,
            examples,
        )

        prompt_len = len(encoded_prompt["input_ids"])
        is_truncated = max_length is not None and prompt_len >= max_length

        if self._arch == "encoder":
            encoded_prompt["labels"] = self.label2id[label]
            encoded_prompt["truncated"] = is_truncated
            return encoded_prompt

        encoded_answer = self._encode_answer(
            system_prompt,
            user_prompt,
            label,
            max_length,
            examples,
        )

        encoded_labels = encoded_answer["input_ids"].copy()
        is_truncated = is_truncated or (
            max_length is not None and len(encoded_labels) >= max_length
        )

        if self._arch == "decoder":
            encoded_labels[:prompt_len] = [SENTINEL_TOKEN] * prompt_len
            encoded_answer["labels"] = encoded_labels
            encoded_answer["truncated"] = is_truncated
            return encoded_answer

        idx = prompt_len - int(encoded_labels[-1] == self._tokenizer.eos_token_id)
        encoded_prompt["labels"] = encoded_labels[idx:]
        encoded_prompt["truncated"] = is_truncated
        return encoded_prompt

    def _sample_examples(self, n_shot: int) -> list[_Example]:
        train = self.raw_data["train"]
        max_examples = len(train)

        if n_shot > max_examples:
            msg = f"requested more than {max_examples} examples"
            raise ValueError(msg)

        if n_shot == 0:
            return []

        examples = train.select(range(n_shot))
        return [
            _Example(user=self._build_user_prompt(example), assistant=example["label"])
            for example in examples
        ]

    @staticmethod
    def _translate_en_to_et(example: _Sample) -> _Sample:
        en_lbl = example["label"]
        example["label"] = Oblique._en_to_et[en_lbl]
        return example
