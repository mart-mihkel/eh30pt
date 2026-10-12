"""Resolved datasets, hyperparameters, and derived loop settings."""

from typing import TYPE_CHECKING, Final, NamedTuple, cast

import torch

from eh30pt.modeling.training.constraints import _label_completions

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch.nn import Module
    from torch.utils.data import IterableDataset as TorchIterableDataset
    from transformers import DataCollator, PreTrainedTokenizerFast, TrainerCallback

    from eh30pt.types import Architecture, ComputeMetricsFn, DataSplits


_no_epoch_cap: Final = 0


class _Spec(NamedTuple):
    """Resolved datasets, hyperparameters, and derived loop settings."""

    model: Module
    tokenizer: PreTrainedTokenizerFast
    train_dataset: TorchIterableDataset | None
    eval_dataset: TorchIterableDataset
    collate_fn: DataCollator
    metrics_fn: Callable
    output_dir: str
    eval_steps: int
    train_steps: int
    learning_rate: float
    batch_size: int
    warmup_ratio: float
    run_name: str
    report_to: str
    checkpoints: bool
    callbacks: list[TrainerCallback]
    optim: str
    eval_strategy: str
    save_strategy: str
    save_steps: int
    load_best: bool
    have_accel: bool
    constrain: bool
    label_completions: tuple[tuple[int, ...], ...]


def _build_spec(
    model: Module,
    *,
    tokenizer: PreTrainedTokenizerFast,
    data: DataSplits,
    arch: Architecture,
    collate_fn: DataCollator,
    metrics_fn: ComputeMetricsFn,
    output_dir: Path | str,
    eval_steps: int,
    train_steps: int = 0,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    warmup_ratio: float = 0.06,
    run_name: str = "default",
    report_to: str = "none",
    checkpoints: bool = False,
    callbacks: tuple[TrainerCallback, ...] = (),
    constrain: bool = False,
) -> _Spec:
    """Resolve datasets, hyperparameters, and derived loop settings."""
    have_cuda = torch.cuda.is_available()
    have_mps = torch.mps.is_available()
    have_accel = have_cuda or have_mps
    do_checkpoints = eval_steps > 0 and checkpoints
    label_completions = (
        _label_completions(data, eos_token_id=tokenizer.eos_token_id)
        if constrain and arch != "encoder"
        else ()
    )

    return _Spec(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data.train,
        eval_dataset=data.eval,
        collate_fn=collate_fn,
        metrics_fn=cast("Callable", metrics_fn),
        output_dir=str(output_dir),
        eval_steps=eval_steps,
        train_steps=train_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        run_name=run_name,
        report_to=report_to,
        checkpoints=checkpoints,
        callbacks=list(callbacks),
        optim="adamw_8bit" if have_cuda else "adamw_torch_fused",
        eval_strategy="steps" if eval_steps > 0 else "no",
        save_strategy="steps" if do_checkpoints else ("epoch" if checkpoints else "no"),
        save_steps=eval_steps if eval_steps > 0 else 500,
        load_best=do_checkpoints,
        have_accel=have_accel,
        constrain=constrain,
        label_completions=label_completions,
    )
