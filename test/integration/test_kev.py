"""Kev-style decision encoding, training, and checkpoint integration tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch

from eh30pt.modeling.kev import (
    ContextOverflowError,
    KevRecord,
    calibrate_kev,
    encode_kev_row,
    encode_records,
    evaluate_kev,
    load_kev_model,
    records_from_split,
)
from eh30pt.modeling.kev.prompt import Delimiters
from eh30pt.modeling.kev.trainer import get_kev_trainer
from eh30pt.settings import LoRASettings

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast

    from eh30pt.types import DataSplits

_rng_seed: Final = 0
_head_dim: Final = 8
_steps_per_epoch: Final = 2
_options: Final = ("alpha", "beta", "gamma")


@pytest.fixture(scope="module")
def delimiters(qwen_tokenizer: PreTrainedTokenizerFast) -> Delimiters:
    return Delimiters.from_tokenizer(qwen_tokenizer)


def test_encode_kev_row_marks_the_pointer_positions(
    qwen_tokenizer: PreTrainedTokenizerFast,
    delimiters: Delimiters,
) -> None:
    encoded = encode_kev_row(
        qwen_tokenizer,
        state="the patient reports a cough",
        instructions="Pick the urgency",
        options=list(_options),
    )

    assert encoded.input_ids[0] == delimiters.state
    assert encoded.input_ids[encoded.decide_index] == delimiters.decide
    assert len(encoded.option_indices) == len(_options)
    assert all(
        encoded.input_ids[index] == delimiters.option_close
        for index in encoded.option_indices
    )
    assert encoded.option_indices == tuple(sorted(encoded.option_indices))
    assert encoded.decide_index == len(encoded.input_ids) - 1
    assert encoded.position_ids == tuple(range(len(encoded.input_ids)))
    assert encoded.state_tokens >= 1
    assert not encoded.truncated


def test_encode_kev_row_rejects_a_state_over_budget(
    qwen_tokenizer: PreTrainedTokenizerFast,
) -> None:
    with pytest.raises(ContextOverflowError, match="state exceeds"):
        encode_kev_row(
            qwen_tokenizer,
            state="word " * 200,
            instructions="Pick one",
            options=list(_options),
            max_state=8,
        )


def test_encode_kev_row_rejects_a_branch_over_budget(
    qwen_tokenizer: PreTrainedTokenizerFast,
) -> None:
    with pytest.raises(ContextOverflowError, match="row exceeds"):
        encode_kev_row(
            qwen_tokenizer,
            state="short",
            instructions="Pick one",
            options=list(_options),
            max_branch=8,
        )


def test_encode_records_drops_rows_over_budget(
    qwen_tokenizer: PreTrainedTokenizerFast,
) -> None:
    records = [
        KevRecord(
            state="short",
            instructions="Pick one",
            options=_options,
            label=0,
        ),
        KevRecord(
            state="word " * 200,
            instructions="Pick one",
            options=_options,
            label=1,
        ),
    ]

    samples = encode_records(qwen_tokenizer, records, max_state=16)

    assert len(samples) == 1
    assert samples[0].label == 0


def test_records_from_split_reuses_dataset_prompts(
    oblique_qwen_dataset: object,
) -> None:
    from eh30pt.datasets.registry import get_labels

    splits = cast("DataSplits", oblique_qwen_dataset)
    labels = get_labels("oblique")
    sampled = list(splits.train.samples())
    assert sampled

    records = records_from_split(
        splits.train,
        instructions="Pick a label",
        labels=labels,
    )

    assert len(records) == len(sampled)
    assert all(record.options == labels for record in records)
    assert all(0 <= record.label < len(labels) for record in records)
    assert records[0].state.endswith(sampled[0].user)


def test_train_and_save_a_checkpoint(
    tmp_path: Path,
    qwen_tokenizer: PreTrainedTokenizerFast,
) -> None:
    torch.manual_seed(_rng_seed)
    records = [
        KevRecord(
            state=f"patient {index} reports symptom {index}",
            instructions="Pick one",
            options=_options,
            label=index % len(_options),
        )
        for index in range(4)
    ]
    samples = encode_records(qwen_tokenizer, records)
    assert len(samples) == len(records)

    model = load_kev_model(
        "tiny-random/qwen3",
        qwen_tokenizer,
        torch.device("cpu"),
        lora=LoRASettings(rank=4, alpha=4, dropout=0.0),
        head_dim=_head_dim,
    )

    trainer = get_kev_trainer(
        model,
        train=samples,
        evaluation=samples,
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_ratio=0.06,
        patience=1,
        tolerance=0.01,
        seed=_rng_seed,
        output_dir=tmp_path / "trainer",
    )
    trainer.train()
    temperature = calibrate_kev(model, samples, batch_size=2)

    assert trainer.state.global_step == _steps_per_epoch
    assert temperature > 0

    metrics = evaluate_kev(
        model,
        samples,
        batch_size=2,
        temperature=temperature,
    )
    assert metrics.count == len(samples)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0

    output = model.save_checkpoint(
        qwen_tokenizer,
        tmp_path / "checkpoint",
        base_model="tiny-random/qwen3",
        base_revision=None,
        lora_rank=4,
        temperature=temperature,
        training_args={"dataset": "oblique", "seed": _rng_seed},
    )

    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "head.pt",
        "training_config.json",
    ):
        assert (output / name).is_file(), name

    config = cast(
        "dict[str, object]",
        json.loads((output / "training_config.json").read_text()),
    )
    args = cast("dict[str, object]", config["args"])
    assert args["head_dim"] == _head_dim
    assert args["dataset"] == "oblique"

    restored = load_kev_model(
        "tiny-random/qwen3",
        qwen_tokenizer,
        torch.device("cpu"),
        lora=LoRASettings(rank=4, alpha=4, dropout=0.0),
        head_dim=_head_dim,
        init_from=str(output),
    )
    assert torch.equal(
        restored.head.q.weight.detach().cpu(),
        model.head.q.weight.detach().cpu(),
    )
