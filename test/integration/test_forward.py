"""Forward-pass integration tests across architectures."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Final, NamedTuple, TypedDict

import pytest
import torch

from eh30pt.constants import PAD_MULTIPLE, SENTINEL_TOKEN
from eh30pt.datasets.discont.extract import DiscontExtract
from eh30pt.datasets.registry import (
    get_extract_fields,
    get_response_schema,
    get_sys_prompt,
)
from eh30pt.modeling.arch import infer_architecture
from eh30pt.modeling.collator import get_collator
from eh30pt.openai.complete import CompletionClient
from eh30pt.settings import ServerSettings

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from eh30pt.types import Architecture
    from test.integration.conftest import ModelSetup

_batch_size: Final = 4


class _Batch(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    truncated: torch.Tensor


class _Inputs(NamedTuple):
    arch: Architecture
    model: PreTrainedModel
    batch: _Batch
    n_labels: int
    tokenizer_vocab_size: int
    model_vocab_size: int


def _model_vocab_size(model: PreTrainedModel) -> int:
    output = model.get_output_embeddings()
    if output is not None:
        return int(output.weight.shape[0])

    config = model.config
    if "text_config" in config:
        config = config.text_config

    return int(config.vocab_size)


@pytest.fixture
def collated_inputs(model_setup: ModelSetup) -> _Inputs:
    """Collate one device-ready batch from the real training split."""
    arch = infer_architecture(model_setup.config)

    rows = []
    for row in model_setup.splits.train:
        rows.append(row)
        if len(rows) == _batch_size:
            break

    raw = get_collator(model_setup.tokenizer, arch)(rows)

    device = next(model_setup.model.parameters()).device
    batch = _Batch(
        input_ids=raw["input_ids"].to(device),
        attention_mask=raw["attention_mask"].to(device),
        labels=raw["labels"].to(device),
        truncated=raw["truncated"].to(device),
    )

    return _Inputs(
        arch=arch,
        model=model_setup.model,
        batch=batch,
        n_labels=len(model_setup.splits.train.id2label),
        tokenizer_vocab_size=len(model_setup.tokenizer),
        model_vocab_size=_model_vocab_size(model_setup.model),
    )


def test_collated_batch(collated_inputs: _Inputs) -> None:
    batch = collated_inputs.batch
    device = next(collated_inputs.model.parameters()).device

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    batch_size, seq_len = input_ids.shape
    assert input_ids.shape == attention_mask.shape
    assert batch_size == _batch_size
    assert seq_len % PAD_MULTIPLE == 0

    assert input_ids.device == device
    assert attention_mask.device == device
    assert labels.device == device

    assert set(attention_mask.unique().tolist()) <= {0, 1}
    assert bool(
        ((input_ids >= 0) & (input_ids < collated_inputs.tokenizer_vocab_size)).all()
    )

    if collated_inputs.arch == "encoder":
        assert tuple(labels.shape) == (_batch_size,)
        assert bool(((labels >= 0) & (labels < collated_inputs.n_labels)).all())
    else:
        label_rows, _ = labels.shape
        assert label_rows == _batch_size

        valid = labels != SENTINEL_TOKEN
        assert bool(valid.any())
        assert bool(
            (
                (labels[valid] >= 0)
                & (labels[valid] < collated_inputs.tokenizer_vocab_size)
            ).all()
        )


def test_forward(collated_inputs: _Inputs) -> None:
    collated_inputs.model.eval()
    with torch.inference_mode():
        output = collated_inputs.model(**collated_inputs.batch)

    loss = output.loss
    assert loss is not None
    assert bool(torch.isfinite(loss))
    assert loss.item() > 0

    logits = output.logits
    assert isinstance(logits, torch.Tensor)
    assert bool(torch.isfinite(logits).all())

    input_ids = collated_inputs.batch["input_ids"]
    label_ids = collated_inputs.batch["labels"]
    batch_size, seq_len = input_ids.shape

    if collated_inputs.arch == "encoder":
        assert tuple(logits.shape) == (batch_size, collated_inputs.n_labels)
    elif collated_inputs.arch == "decoder":
        assert tuple(logits.shape) == (
            batch_size,
            seq_len,
            collated_inputs.model_vocab_size,
        )
    elif collated_inputs.arch == "encoder-decoder":
        assert tuple(logits.shape) == (
            batch_size,
            label_ids.shape[1],
            collated_inputs.model_vocab_size,
        )


@pytest.mark.vcr
def test_extract_constrained_forward() -> None:
    """Constrained decoding forward yields parsed JSON, plain forward text."""
    user = DiscontExtract.extract_shots[0].user
    conversation = CompletionClient.build_conversation(
        user, system=get_sys_prompt("discont-extract/trigger")
    )

    constrained = CompletionClient(
        ServerSettings(
            provider="llama",
            format=get_response_schema("discont-extract/trigger"),
        ),
        seed=0,
    )
    parsed = asyncio.run(constrained.complete(conversation)).unwrap()

    assert isinstance(parsed, dict)
    assert set(parsed) == set(get_extract_fields("discont-extract/trigger"))

    plain = CompletionClient(ServerSettings(provider="llama"), seed=0)
    text = asyncio.run(plain.complete(conversation)).unwrap()

    assert isinstance(text, str)
    assert text.strip()
