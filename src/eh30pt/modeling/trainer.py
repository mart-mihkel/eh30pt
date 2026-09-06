"""Trainer construction, training arguments, and seq2seq quirks."""

from typing import TYPE_CHECKING, cast

import torch
from transformers import (
    DataCollator,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from datasets.dataset_dict import DatasetDict
    from torch.nn import Module
    from torch.utils.data import Dataset

    from eh30pt.types import Architecture, ComputeMetricsFn


def get_args(
    arch: Architecture,
    output_dir: Path | str,
    *,
    do_eval: bool,
    epochs: int = 0,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    run_name: str = "default",
    report_to: str = "none",
    checkpoints: bool = False,
) -> TrainingArguments:
    """
    Build Trainer arguments, using seq2seq args for encoder-decoder models.

    Parameters
    ----------
    arch:
        Model architecture, selecting regular or seq2seq arguments.
    output_dir:
        Directory for checkpoints and trainer state.
    do_eval:
        Run evaluation during training.
    epochs:
        Number of training epochs.
    learning_rate:
        Optimizer learning rate.
    batch_size:
        Training and evaluation batch size.
    run_name:
        Display name reported to trackers.
    report_to:
        Tracker integrations enabled on the Trainer.
    checkpoints:
        Save epoch checkpoints (also required to restore the best model).

    Returns
    -------
    TrainingArguments
        Arguments for the matching Trainer flavor.

    """
    have_cuda = torch.cuda.is_available()
    have_mps = torch.mps.is_available()
    have_accel = have_cuda or have_mps
    optim = "adamw_8bit" if have_cuda else "adamw_torch_fused"
    eval_strategy = "epoch" if do_eval else "no"
    save_strategy = "epoch" if checkpoints else "no"
    # Restoring the best checkpoint requires periodic saves to compare.
    load_best = do_eval and checkpoints

    if arch == "encoder-decoder":
        logger.debug("using seq2seq training args")
        args = Seq2SeqTrainingArguments(
            run_name=run_name,
            report_to=report_to,
            output_dir=str(output_dir),
            save_strategy=save_strategy,
            save_total_limit=1,
            eval_strategy=eval_strategy,
            eval_on_start=do_eval,
            batch_eval_metrics=True,
            remove_unused_columns=False,
            logging_steps=100,
            load_best_model_at_end=load_best,
            metric_for_best_model="f1",
            learning_rate=learning_rate,
            optim=optim,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            use_cpu=not have_accel,
            bf16_full_eval=have_accel,
            bf16=have_accel,
            predict_with_generate=True,
            generation_max_length=8,
        )
    else:
        logger.debug("using regular training args")
        args = TrainingArguments(
            run_name=run_name,
            report_to=report_to,
            output_dir=str(output_dir),
            save_strategy=save_strategy,
            save_total_limit=1,
            eval_strategy=eval_strategy,
            eval_on_start=do_eval,
            batch_eval_metrics=True,
            remove_unused_columns=False,
            logging_steps=100,
            load_best_model_at_end=load_best,
            metric_for_best_model="f1",
            learning_rate=learning_rate,
            optim=optim,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            use_cpu=not have_accel,
            bf16_full_eval=have_accel,
            bf16=have_accel,
        )

    return args


def get_trainer(
    model: Module,
    *,
    data: DatasetDict,
    arch: Architecture,
    collate_fn: DataCollator,
    metrics_fn: ComputeMetricsFn,
    output_dir: Path | str,
    do_eval: bool,
    patience: int = 0,
    tolerance: float = 0.01,
    epochs: int = 0,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    run_name: str = "default",
    report_to: str = "none",
    checkpoints: bool = False,
    callbacks: tuple[TrainerCallback, ...] = (),
) -> Trainer:
    """
    Build a Trainer for the architecture, adding early stopping if requested.

    Parameters
    ----------
    model:
        Model to train or evaluate.
    data:
        Tokenized dataset splits (`train`, `validation`).
    arch:
        Model architecture, selecting regular or seq2seq Trainer.
    collate_fn:
        Data collator batching tokenized examples.
    metrics_fn:
        Metric function scoring evaluation predictions.
    output_dir:
        Directory for checkpoints and trainer state.
    do_eval:
        Run evaluation during training.
    patience:
        Eval epochs without improvement before stopping, 0 disables.
    tolerance:
        Minimum eval improvement resetting the patience counter.
    epochs:
        Number of training epochs.
    learning_rate:
        Optimizer learning rate.
    batch_size:
        Training and evaluation batch size.
    run_name:
        Display name reported to trackers.
    report_to:
        Tracker integrations enabled on the Trainer.
    checkpoints:
        Save epoch checkpoints (also required to restore the best model).
    callbacks:
        Extra Trainer callbacks.

    Returns
    -------
    Trainer
        Configured Trainer for the architecture.

    """
    args = get_args(
        arch,
        output_dir,
        do_eval=do_eval,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        run_name=run_name,
        report_to=report_to,
        checkpoints=checkpoints,
    )

    _metrics_fn = cast("Callable", metrics_fn)
    train_dataset = data.get("train")
    eval_dataset = cast("Dataset", data.get("validation"))

    if arch == "encoder-decoder":
        logger.debug("using seq2seq trainer")
        trainer_cls = Seq2SeqTrainer
    else:
        logger.debug("using regular trainer")
        trainer_cls = Trainer

    trainer = trainer_cls(
        args=args,
        model=model,
        data_collator=collate_fn,
        eval_dataset=eval_dataset,
        train_dataset=train_dataset,
        compute_metrics=_metrics_fn,
        callbacks=list(callbacks),
    )

    if patience > 0 and not do_eval:
        logger.warning("not using early stopping because not running evaluation")
    elif do_eval and patience > 0:
        logger.info(
            "using early stopping with %d patience and %.2f tolerance",
            patience,
            tolerance,
        )

        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=tolerance,
            )
        )

    return trainer
