"""MultiNERD dataset loading, prompting, and tokenization."""

from textwrap import dedent
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, TypedDict, cast

from datasets.load import load_dataset
from datasets.utils.info_utils import VerificationMode

from eh30pt.constants import SENTINEL_TOKEN
from eh30pt.datasets.truncation import get_max_length
from eh30pt.logging import logger
from eh30pt.types import DatasetProtocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from datasets.dataset_dict import DatasetDict
    from datasets.splits import Split
    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.types import Architecture

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

type _Tag = Literal[
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


class _Batch(TypedDict):
    tokens: list[list[str]]
    ner_tags: list[list[int]]
    lang: list[_Lang]


class _TokBatch(TypedDict):
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    labels: list[int | list[int]]
    truncated: list[bool]


class _Example(NamedTuple):
    user: str
    assistant: str


class Multinerd(DatasetProtocol):
    """
    MultiNERD NER tag classification dataset.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerFast
        Tokenizer used for prompting and tokenization.
    arch : Architecture
        Model architecture, selecting the prompt format.

    """

    id2label: ClassVar[Mapping[int, _Tag]] = {
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

    label2id: ClassVar[Mapping[_Tag, int]] = {
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

    _id2label_bio: ClassVar[Mapping[int, str]] = {
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
                "What is the NER tag of the entity in the sentence?"
                f"{self._tokenizer.sep_token}"
            )

        return dedent(f"""
            Identify the NER tag of the entity in the sentence.
            Possible tags are: {", ".join(self.id2label.values())}.

            Output only the tag.
        """).strip()

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
        filter_en: bool = True,
    ) -> None:
        """
        Load a modified subset of the MultiNERD dataset into instance state.

        The BIO tagging task is converted to a regular NER tagging task by
        joining tokens with B- and I- prefixes into a single span. O tags are
        dropped entirely.

        Each token is split into a separate sample containing the entire
        context sentence and the target token. The task is to classify the tag
        of the token in the entire sequence.

        Parameters
        ----------
        n_shot : int
            Number of few-shot examples prepended to each prompt.
        n_virtual : int
            Number of virtual tokens reserved in the truncation budget.
        split : Split | None
            Dataset split to load, or None for all splits.
        filter_en : bool
            Only use the english subset.

        """
        self._raw_data = self._load_raw(split=split, filter_en=filter_en)
        self._data = self._tokenize(n_shot, n_virtual)

    def _load_raw(self, split: Split | None, *, filter_en: bool) -> DatasetDict:
        logger.debug("loading raw multinerd dataset")
        data = load_dataset(
            "Babelscape/multinerd",
            split=split,
            verification_mode=VerificationMode.NO_CHECKS,
        )

        data = cast("DatasetDict", data)
        if filter_en:
            logger.warning("using english only subset")
            data = data.filter(self._filter_english, batched=True)

        return data

    def _tokenize(self, n_shot: int, n_virtual: int, num_proc: int = 4) -> DatasetDict:
        logger.debug("tokenizing multinerd")

        examples = self._sample_examples(n_shot)
        data = self.raw_data.map(
            self._tokenize_batch,
            batched=True,
            num_proc=num_proc,
            remove_columns=["tokens", "ner_tags", "lang"],
            fn_kwargs={"examples": examples, "n_virtual": n_virtual},
        )

        for subsplit in data:
            logger.debug("tokenized %d %s samples", len(data[subsplit]), subsplit)

        return data

    def _build_user_prompt(self, sentence: str, entity: str) -> str:
        if self._arch == "encoder":
            return f"{sentence}{self._tokenizer.sep_token}{entity}"

        return dedent(f"""
            Sentence: {sentence}
            Entity: {entity}
            Tag:
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
        tag_id: int,
        max_length: int | None,
        examples: list[_Example],
    ) -> BatchEncoding[list[int]]:
        """Encode the prompt together with the gold label."""
        if self._tokenizer.chat_template is None:
            examples_prompt = self._format_examples_legacy(examples)
            user_prompt = f"{examples_prompt}\n{user_prompt}"
            encoded_answer = self._tokenizer(
                f"{system_prompt}\n{user_prompt} {self.id2label[tag_id]}",
                truncation=True,
                max_length=max_length,
            )
        else:
            conversation: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt}
            ]

            conversation.extend(self._format_examples_template(examples))
            conversation.append({"role": "user", "content": user_prompt})
            conversation.append({"role": "assistant", "content": self.id2label[tag_id]})

            encoded_answer = self._tokenizer.apply_chat_template(
                conversation,
                truncation=True,
                max_length=max_length,
                return_dict=True,
            )

        return cast("BatchEncoding[list[int]]", encoded_answer)

    def _tokenize_batch(
        self,
        batch: _Batch,
        examples: list[_Example],
        n_virtual: int = 0,
    ) -> _TokBatch:
        all_ids: list[list[int]] = []
        all_attn: list[list[int]] = []
        all_labels: list[int | list[int]] = []
        all_truncated: list[bool] = []
        max_length = get_max_length(self._tokenizer, n_virtual)

        system_prompt = self.system_prompt
        token_tags = zip(batch["tokens"], batch["ner_tags"], strict=True)
        for tokens, raw_tag_ids in token_tags:
            sentence = " ".join(tokens)
            entities, tag_ids = self._join_spans(tokens, raw_tag_ids)

            for entity, tag_id in zip(entities, tag_ids, strict=True):
                if tag_id == -1:
                    continue

                user_prompt = self._build_user_prompt(sentence, entity)
                encoded_prompt = self._encode_prompt(
                    system_prompt,
                    user_prompt,
                    max_length,
                    examples,
                )

                prompt_len = len(encoded_prompt["input_ids"])
                is_truncated = max_length is not None and prompt_len >= max_length

                if self._arch == "encoder":
                    all_ids.append(encoded_prompt["input_ids"])
                    all_attn.append(encoded_prompt["attention_mask"])
                    all_labels.append(tag_id)
                    all_truncated.append(is_truncated)
                    continue

                encoded_answer = self._encode_answer(
                    system_prompt,
                    user_prompt,
                    tag_id,
                    max_length,
                    examples,
                )

                encoded_labels = encoded_answer["input_ids"].copy()
                is_truncated = is_truncated or (
                    max_length is not None and len(encoded_labels) >= max_length
                )

                if self._arch == "decoder":
                    encoded_labels[:prompt_len] = [SENTINEL_TOKEN] * prompt_len

                    all_ids.append(encoded_answer["input_ids"])
                    all_attn.append(encoded_answer["attention_mask"])
                    all_labels.append(encoded_labels)
                    all_truncated.append(is_truncated)
                    continue

                if self._arch == "encoder-decoder":
                    idx = prompt_len - int(
                        encoded_labels[-1] == self._tokenizer.eos_token_id
                    )

                    all_ids.append(encoded_prompt["input_ids"])
                    all_attn.append(encoded_prompt["attention_mask"])
                    all_labels.append(encoded_labels[idx:])
                    all_truncated.append(is_truncated)
                    continue

        return _TokBatch(
            input_ids=all_ids,
            attention_mask=all_attn,
            labels=all_labels,
            truncated=all_truncated,
        )

    def _sample_examples(self, n_shot: int) -> list[_Example]:
        if n_shot == 0:
            return []

        examples: list[_Example] = []
        for sample in self.raw_data["train"]:
            tokens = cast("list[str]", sample["tokens"])
            raw_tag_ids = cast("list[int]", sample["ner_tags"])
            sentence = " ".join(tokens)
            entities, tag_ids = Multinerd._join_spans(tokens, raw_tag_ids)

            for entity, tag_id in zip(entities, tag_ids, strict=True):
                if tag_id == -1:
                    continue

                example = _Example(
                    user=self._build_user_prompt(sentence, entity),
                    assistant=Multinerd.id2label[tag_id],
                )

                examples.append(example)
                break

            if len(examples) == n_shot:
                return examples

        msg = f"requested more than {len(examples)} examples"
        raise ValueError(msg)

    @staticmethod
    def _join_spans(
        tokens: list[str],
        tag_ids: list[int],
    ) -> tuple[list[str], list[int]]:
        out_ids = []
        out_tokens = []
        for token, tag_id in zip(tokens, tag_ids, strict=True):
            tag = Multinerd._id2label_bio[tag_id]

            if tag.startswith("B-"):
                tag = cast("_Tag", tag[2:])
                out_ids.append(Multinerd.label2id[tag])
                out_tokens.append(token)
            elif tag.startswith("I-"):
                out_tokens[-1] = f"{out_tokens[-1]} {token}"
            elif tag == "O":
                out_ids.append(-1)
                out_tokens.append(token)
            else:
                tag = cast("_Tag", tag)
                out_ids.append(Multinerd.label2id[tag])
                out_tokens.append(token)

        return out_tokens, out_ids

    @staticmethod
    def _filter_english(batch: _Batch) -> list[bool]:
        return [lang == "en" for lang in batch["lang"]]
