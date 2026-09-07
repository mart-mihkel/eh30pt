"""Forward-pass integration tests across architectures."""

from typing import TYPE_CHECKING, Final, NamedTuple, TypedDict

import torch

from eh30pt.constants import PAD_MULTIPLE, SENTINEL_TOKEN
from eh30pt.modeling.arch import infer_architecture
from eh30pt.modeling.collator import get_collator

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from eh30pt.types import Architecture
    from test.conftest import ModelSetup

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
    vocab_size: int


def _model_device(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


def _to_device(value: object, device: torch.device) -> torch.Tensor:
    assert isinstance(value, torch.Tensor)
    return value.to(device)


def _collate(model_setup: ModelSetup) -> _Inputs:
    arch = infer_architecture(model_setup.config)

    rows = model_setup.dataset.data["train"].select(range(_batch_size)).to_list()
    raw = get_collator(model_setup.tokenizer, arch)(rows)

    device = _model_device(model_setup.model)
    batch = _Batch(
        input_ids=_to_device(raw["input_ids"], device),
        attention_mask=_to_device(raw["attention_mask"], device),
        labels=_to_device(raw["labels"], device),
        truncated=_to_device(raw["truncated"], device),
    )

    return _Inputs(
        arch=arch,
        model=model_setup.model,
        batch=batch,
        n_labels=len(model_setup.dataset.id2label),
        vocab_size=len(model_setup.tokenizer),
    )


def test_collated_batch(model_setup: ModelSetup) -> None:
    inputs = _collate(model_setup)
    batch = inputs.batch
    device = _model_device(inputs.model)

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
    assert bool(((input_ids >= 0) & (input_ids < inputs.vocab_size)).all())

    if inputs.arch == "encoder":
        assert tuple(labels.shape) == (_batch_size,)
        assert bool(((labels >= 0) & (labels < inputs.n_labels)).all())
    else:
        label_rows, _ = labels.shape
        assert label_rows == _batch_size

        valid = labels != SENTINEL_TOKEN
        assert bool(valid.any())
        assert bool(((labels[valid] >= 0) & (labels[valid] < inputs.vocab_size)).all())


def test_forward(model_setup: ModelSetup) -> None:
    inputs = _collate(model_setup)
    inputs.model.eval()
    with torch.inference_mode():
        output = inputs.model(**inputs.batch)

    loss = output.loss
    assert loss is not None
    assert bool(torch.isfinite(loss))
    assert loss.item() > 0

    logits = output.logits
    assert isinstance(logits, torch.Tensor)
    assert bool(torch.isfinite(logits).all())

    input_ids = inputs.batch["input_ids"]
    label_ids = inputs.batch["labels"]
    batch_size, seq_len = input_ids.shape

    if inputs.arch == "encoder":
        assert tuple(logits.shape) == (batch_size, inputs.n_labels)
    elif inputs.arch == "decoder":
        assert tuple(logits.shape) == (batch_size, seq_len, inputs.vocab_size)
    elif inputs.arch == "encoder-decoder":
        assert tuple(logits.shape) == (
            batch_size,
            label_ids.shape[1],
            inputs.vocab_size,
        )
