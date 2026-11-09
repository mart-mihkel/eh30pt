"""Resolved datasets, hyperparameters, and derived loop settings."""

from typing import TYPE_CHECKING, Final, NamedTuple, cast

import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch.nn import Module
    from transformers import DataCollator, PreTrainedTokenizerFast, TrainerCallback

    from eh30pt.datasets.mixin import DatasetMixin
    from eh30pt.types import ComputeMetricsFn, DataSplits, EvalSteps


_no_epoch_cap: Final = 0


class _Spec(NamedTuple):
    """Resolved datasets, hyperparameters, and derived loop settings."""

    model: Module
    tokenizer: PreTrainedTokenizerFast
    train_dataset: DatasetMixin | None
    eval_dataset: DatasetMixin
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
    callbacks: list[TrainerCallback]
    optim: str
    eval_strategy: str
    save_strategy: str
    save_steps: int
    have_accel: bool


def _build_spec(
    model: Module,
    *,
    tokenizer: PreTrainedTokenizerFast,
    data: DataSplits,
    collate_fn: DataCollator,
    metrics_fn: ComputeMetricsFn,
    output_dir: Path | str,
    eval_steps: EvalSteps,
    train_steps: int = 0,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    warmup_ratio: float = 0.06,
    run_name: str = "default",
    report_to: str = "none",
    checkpoints: bool = False,
    callbacks: tuple[TrainerCallback, ...] = (),
) -> _Spec:
    """Resolve datasets, hyperparameters, and derived loop settings."""
    have_cuda = torch.cuda.is_available()
    have_mps = torch.mps.is_available()
    have_accel = have_cuda or have_mps
    every_epoch = eval_steps == "epoch"
    every_steps = isinstance(eval_steps, int) and eval_steps > 0
    hf_eval_steps = eval_steps if isinstance(eval_steps, int) else 500

    save_strategy = (
        "steps" if every_steps and checkpoints else ("epoch" if checkpoints else "no")
    )

    return _Spec(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data.train,
        eval_dataset=data.eval,
        collate_fn=collate_fn,
        metrics_fn=cast("Callable", metrics_fn),
        output_dir=str(output_dir),
        eval_steps=hf_eval_steps,
        train_steps=train_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        run_name=run_name,
        report_to=report_to,
        callbacks=list(callbacks),
        optim="adamw_8bit" if have_cuda else "adamw_torch_fused",
        eval_strategy="epoch" if every_epoch else ("steps" if every_steps else "no"),
        save_strategy=save_strategy,
        save_steps=hf_eval_steps if every_steps else 500,
        have_accel=have_accel,
    )
