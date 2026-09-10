"""Streaming dataset interface with few-shot prompt encoding."""

from typing import TYPE_CHECKING, ClassVar, cast

from eh30pt.constants import PAD_MULTIPLE, SENTINEL_TOKEN, UNSET_MAX_LENGTH

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from typing import NoReturn

    from openai.types.shared_params import ResponseFormatJSONSchema
    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from eh30pt.types import Architecture, DatasetTask, ExtractShot, FewShot


class DatasetMixin[RowType]:
    """Streaming dataset with prompt encoding helpers."""

    id2label: ClassVar[Mapping[int, str]]
    label2id: ClassVar[Mapping[str, int]]
    few_shots: ClassVar[list[FewShot]]
    extract_shots: ClassVar[list[ExtractShot]] = []
    system_prompt: ClassVar[str]
    task: ClassVar[DatasetTask] = "classification"

    @classmethod
    def response_format(cls) -> ResponseFormatJSONSchema:
        """Build the response format constraining generation to labels."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "label",
                "schema": {"type": "string", "enum": list(cls.id2label.values())},
            },
        }

    def tokenize(
        self,
        row: RowType,
        tokenizer: PreTrainedTokenizerFast,
        n_shot: int = 0,
        n_virtual: int = 0,
    ) -> BatchEncoding[list[int]]:
        """Tokenize one row with few-shot context."""
        del row, tokenizer, n_shot, n_virtual
        raise NotImplementedError

    def samples(self) -> Iterator[FewShot]:
        """Yield decoder prompt and label pairs for the split."""
        raise NotImplementedError

    def set_epoch(self, epoch: int) -> None:
        """Ignore the epoch counter, rows stream in split order."""
        del epoch

    def __getitem__(self, *args: str | int, **kwargs: str | int) -> NoReturn:
        """Forbid indexing, rows only stream in split order."""
        del args, kwargs
        msg = "streaming datasets do not support indexing, iterate instead"
        raise TypeError(msg)

    def _format_few_shot_legacy(self, few_shots: list[FewShot]) -> str:
        if not few_shots:
            return ""

        return "\n".join(f"{shot.user} {shot.label}\n" for shot in few_shots)

    def _format_few_shot_chat(
        self,
        tokenizer: PreTrainedTokenizerFast,
        few_shots: list[FewShot],
    ) -> list[dict[str, str]]:
        if tokenizer.chat_template is None:
            msg = "attempt to format with no chat template, use _format_examples_legacy"
            raise RuntimeError(msg)

        conversation: list[dict[str, str]] = []
        for shot in few_shots:
            conversation.append({"role": "user", "content": shot.user})
            conversation.append({"role": "assistant", "content": shot.label})

        return conversation

    def _encode_prompt(
        self,
        *,
        arch: Architecture,
        tokenizer: PreTrainedTokenizerFast,
        user_prompt: str,
        few_shots: list[FewShot],
        max_len: int | None,
    ) -> BatchEncoding[list[int]]:
        if tokenizer.chat_template is None:
            examples_prompt = self._format_few_shot_legacy(few_shots)
            user_prompt = f"{examples_prompt}\n{user_prompt}"
            encoded_prompt = tokenizer(
                f"{self.system_prompt}\n{user_prompt}",
                max_length=max_len,
                truncation=True,
            )
        else:
            conversation: list[dict[str, str]] = [
                {"role": "system", "content": self.system_prompt}
            ]

            conversation.extend(self._format_few_shot_chat(tokenizer, few_shots))
            conversation.append({"role": "user", "content": user_prompt})

            encoded_prompt = tokenizer.apply_chat_template(
                conversation,
                truncation=True,
                return_dict=True,
                max_length=max_len,
                add_generation_prompt=arch != "encoder",
            )

        return cast("BatchEncoding[list[int]]", encoded_prompt)

    def _encode_answer(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        user_prompt: str,
        label: str,
        few_shots: list[FewShot],
        max_len: int | None,
    ) -> BatchEncoding[list[int]]:
        """Encode the prompt together with the gold label."""
        if tokenizer.chat_template is None:
            examples_prompt = self._format_few_shot_legacy(few_shots)
            user_prompt = f"{examples_prompt}\n{user_prompt}"
            encoded_answer = tokenizer(
                f"{self.system_prompt}\n{user_prompt} {label}",
                max_length=max_len,
                truncation=True,
            )
        else:
            conversation: list[dict[str, str]] = [
                {"role": "system", "content": self.system_prompt}
            ]

            conversation.extend(self._format_few_shot_chat(tokenizer, few_shots))
            conversation.append({"role": "user", "content": user_prompt})
            conversation.append({"role": "assistant", "content": label})

            encoded_answer = tokenizer.apply_chat_template(
                conversation,
                truncation=True,
                return_dict=True,
                max_length=max_len,
            )

        return cast("BatchEncoding[list[int]]", encoded_answer)

    def _encode_example(
        self,
        *,
        arch: Architecture,
        tokenizer: PreTrainedTokenizerFast,
        user_prompt: str,
        label: str,
        few_shots: list[FewShot],
        n_virtual: int = 0,
    ) -> BatchEncoding[list[int]]:
        """Encode one labeled example into prompt tokens with labels set."""
        max_len = self._model_max_len(tokenizer, n_virtual)
        encoded_prompt = self._encode_prompt(
            arch=arch,
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            few_shots=few_shots,
            max_len=max_len,
        )

        return self._finalize_encoding(
            arch=arch,
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            label=label,
            few_shots=few_shots,
            max_len=max_len,
            encoded_prompt=encoded_prompt,
        )

    def _finalize_encoding(
        self,
        *,
        arch: Architecture,
        tokenizer: PreTrainedTokenizerFast,
        user_prompt: str,
        label: str,
        few_shots: list[FewShot],
        max_len: int | None,
        encoded_prompt: BatchEncoding[list[int]],
    ) -> BatchEncoding[list[int]]:
        """Attach labels to a prompt encoding per architecture."""
        prompt_len = len(encoded_prompt["input_ids"])
        is_truncated = max_len is not None and prompt_len >= max_len

        if arch == "encoder":
            encoded_prompt["labels"] = self.label2id[label]
            encoded_prompt["truncated"] = is_truncated
            return encoded_prompt

        encoded_answer = self._encode_answer(
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            label=label,
            few_shots=few_shots,
            max_len=max_len,
        )

        encoded_labels = encoded_answer["input_ids"].copy()
        is_truncated = is_truncated or (
            max_len is not None and len(encoded_labels) >= max_len
        )

        if arch == "decoder":
            encoded_labels[:prompt_len] = [SENTINEL_TOKEN] * prompt_len
            encoded_answer["labels"] = encoded_labels
            encoded_answer["truncated"] = is_truncated
            return encoded_answer

        idx = prompt_len - int(encoded_labels[-1] == tokenizer.eos_token_id)
        encoded_prompt["labels"] = encoded_labels[idx:]
        encoded_prompt["truncated"] = is_truncated
        return encoded_prompt

    @staticmethod
    def _model_max_len(
        tokenizer: PreTrainedTokenizerFast,
        n_virtual: int,
    ) -> int | None:
        """Return the truncation length that leaves room for virtual tokens."""
        if tokenizer.model_max_length >= UNSET_MAX_LENGTH:
            return None

        budget = tokenizer.model_max_length - n_virtual
        return (budget // PAD_MULTIPLE) * PAD_MULTIPLE
