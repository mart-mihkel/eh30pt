"""Per-architecture Trainer construction."""

from typing import TYPE_CHECKING, Any, Final, NamedTuple, cast

import torch
from transformers import (
    DataCollator,
    EarlyStoppingCallback,
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from trl import SFTConfig, SFTTrainer

from eh30pt.constants import SENTINEL_TOKEN
from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from datasets.iterable_dataset import IterableDataset as HFIterableDataset
    from peft import PeftModel
    from torch.nn import Module
    from torch.utils.data import IterableDataset as TorchIterableDataset
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

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
    label_completions: tuple[tuple[int, ...], ...]


class ConstrainedSFTTrainer(SFTTrainer):
    """
    SFTTrainer constraining eval predictions to legal label token sequences.

    Each row's argmax is forced to its most likely legal completion, scored
    by teacher-forced likelihood over same-length candidates.
    """

    def __init__(
        self,
        *,
        model: PreTrainedModel | PeftModel,
        args: SFTConfig,
        data_collator: DataCollator,
        train_dataset: TorchIterableDataset | None,
        eval_dataset: TorchIterableDataset | None,
        processing_class: PreTrainedTokenizerFast,
        compute_metrics: Callable,
        callbacks: list[TrainerCallback],
        label_completions: tuple[tuple[int, ...], ...],
    ) -> None:
        """Store the legal completions and initialize the SFT trainer."""
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=cast("HFIterableDataset | None", train_dataset),
            eval_dataset=cast("HFIterableDataset | None", eval_dataset),
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        self._completions = label_completions

    def prediction_step(
        self,
        model: Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,  # noqa: FBT001
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Constrain eval logits to legal label continuations."""
        loss, logits, labels = super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )

        if prediction_loss_only or logits is None or labels is None:
            return loss, logits, labels

        if isinstance(logits, tuple):
            logits = logits[0]

        return loss, _constrain_eval_logits(logits, labels, self._completions), labels


def _encoder_trainer(spec: _Spec) -> Trainer:
    """Build the classification Trainer."""
    logger.debug("using regular trainer")
    args = TrainingArguments(
        run_name=spec.run_name,
        report_to=spec.report_to,
        output_dir=spec.output_dir,
        save_strategy=spec.save_strategy,
        save_steps=spec.save_steps,
        save_total_limit=1,
        eval_strategy=spec.eval_strategy,
        eval_steps=spec.eval_steps,
        eval_on_start=spec.eval_steps > 0,
        batch_eval_metrics=True,
        remove_unused_columns=False,
        logging_steps=100,
        load_best_model_at_end=spec.load_best,
        metric_for_best_model="f1",
        learning_rate=spec.learning_rate,
        warmup_steps=spec.warmup_ratio,
        optim=spec.optim,
        num_train_epochs=_no_epoch_cap,
        max_steps=spec.train_steps,
        per_device_train_batch_size=spec.batch_size,
        per_device_eval_batch_size=spec.batch_size,
        use_cpu=not spec.have_accel,
        bf16_full_eval=spec.have_accel,
        bf16=spec.have_accel,
    )

    return Trainer(
        args=args,
        model=spec.model,
        data_collator=spec.collate_fn,
        eval_dataset=spec.eval_dataset,
        train_dataset=spec.train_dataset,
        compute_metrics=spec.metrics_fn,
        callbacks=spec.callbacks,
    )


def _decoder_trainer(spec: _Spec) -> Trainer:
    """Build the label-constrained SFT Trainer for causal models."""
    logger.debug("using constrained SFT trainer")
    args = SFTConfig(
        run_name=spec.run_name,
        report_to=spec.report_to,
        output_dir=spec.output_dir,
        save_strategy=spec.save_strategy,
        save_steps=spec.save_steps,
        save_total_limit=1,
        eval_strategy=spec.eval_strategy,
        eval_steps=spec.eval_steps,
        eval_on_start=spec.eval_steps > 0,
        batch_eval_metrics=True,
        remove_unused_columns=False,
        logging_steps=100,
        load_best_model_at_end=spec.load_best,
        metric_for_best_model="f1",
        learning_rate=spec.learning_rate,
        warmup_steps=spec.warmup_ratio,
        optim=spec.optim,
        num_train_epochs=_no_epoch_cap,
        max_steps=spec.train_steps,
        per_device_train_batch_size=spec.batch_size,
        per_device_eval_batch_size=spec.batch_size,
        use_cpu=not spec.have_accel,
        bf16_full_eval=spec.have_accel,
        bf16=spec.have_accel,
        dataset_kwargs={"skip_prepare_dataset": True},
        packing=False,
        loss_type="nll",
    )

    return ConstrainedSFTTrainer(
        model=cast("PreTrainedModel | PeftModel", spec.model),
        args=args,
        data_collator=spec.collate_fn,
        train_dataset=spec.train_dataset,
        eval_dataset=spec.eval_dataset,
        processing_class=spec.tokenizer,
        compute_metrics=spec.metrics_fn,
        callbacks=spec.callbacks,
        label_completions=spec.label_completions,
    )


class ConstrainedSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer constraining generation to legal label token sequences.

    Generation is masked through `PrefixConstrainedLogitsProcessor` to tokens
    continuing the decoded prefix toward a legal label.
    """

    def __init__(
        self,
        *,
        model: Module,
        args: Seq2SeqTrainingArguments,
        data_collator: DataCollator,
        train_dataset: TorchIterableDataset | None,
        eval_dataset: TorchIterableDataset,
        compute_metrics: Callable,
        callbacks: list[TrainerCallback],
        label_completions: tuple[tuple[int, ...], ...],
        pad_token_id: int,
    ) -> None:
        """Build the prefix processor and initialize the seq2seq trainer."""
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        def allowed(_batch_id: int, sent: torch.Tensor) -> list[int]:
            """Return legal continuations past the decoder start token."""
            tail: tuple[int, ...] = tuple(sent[1:].tolist())
            return _allowed_next(label_completions, pad_token_id, tail)

        self._gen_kwargs: dict[str, Any] = {}
        self._logits_processor = LogitsProcessorList(
            [PrefixConstrainedLogitsProcessor(allowed, num_beams=1)]
        )

    def prediction_step(
        self,
        model: Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,  # noqa: FBT001
        ignore_keys: list[str] | None = None,
        **gen_kwargs: Any,  # noqa: ANN401
    ) -> tuple[float | None, torch.Tensor | None, torch.Tensor | None]:
        """Generate with the label-constraining logits processor injected."""
        inputs = {k: v for k, v in inputs.items() if k != "truncated"}
        merged = {**self._gen_kwargs, **gen_kwargs}
        merged.setdefault("logits_processor", self._logits_processor)

        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
            **merged,
        )


def _seq2seq_trainer(spec: _Spec) -> Trainer:
    """Build the label-constrained seq2seq Trainer for encoder-decoder models."""
    logger.debug("using constrained seq2seq trainer")
    args = Seq2SeqTrainingArguments(
        run_name=spec.run_name,
        report_to=spec.report_to,
        output_dir=spec.output_dir,
        save_strategy=spec.save_strategy,
        save_steps=spec.save_steps,
        save_total_limit=1,
        eval_strategy=spec.eval_strategy,
        eval_steps=spec.eval_steps,
        eval_on_start=spec.eval_steps > 0,
        batch_eval_metrics=True,
        remove_unused_columns=False,
        logging_steps=100,
        load_best_model_at_end=spec.load_best,
        metric_for_best_model="f1",
        learning_rate=spec.learning_rate,
        warmup_steps=spec.warmup_ratio,
        optim=spec.optim,
        num_train_epochs=_no_epoch_cap,
        max_steps=spec.train_steps,
        per_device_train_batch_size=spec.batch_size,
        per_device_eval_batch_size=spec.batch_size,
        use_cpu=not spec.have_accel,
        bf16_full_eval=spec.have_accel,
        bf16=spec.have_accel,
        predict_with_generate=True,
        generation_max_length=8,
    )

    return ConstrainedSeq2SeqTrainer(
        model=spec.model,
        args=args,
        data_collator=spec.collate_fn,
        train_dataset=spec.train_dataset,
        eval_dataset=spec.eval_dataset,
        compute_metrics=spec.metrics_fn,
        callbacks=spec.callbacks,
        label_completions=spec.label_completions,
        pad_token_id=cast("int", spec.tokenizer.pad_token_id),
    )


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
) -> _Spec:
    """Resolve datasets, hyperparameters, and derived loop settings."""
    have_cuda = torch.cuda.is_available()
    have_mps = torch.mps.is_available()
    have_accel = have_cuda or have_mps
    do_checkpoints = eval_steps > 0 and checkpoints

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
        label_completions=_label_completions(data, eos_token_id=tokenizer.eos_token_id)
        if arch != "encoder"
        else (),
    )


def get_trainer(
    model: Module,
    *,
    tokenizer: PreTrainedTokenizerFast,
    data: DataSplits,
    arch: Architecture,
    collate_fn: DataCollator,
    metrics_fn: ComputeMetricsFn,
    output_dir: Path | str,
    eval_steps: int,
    patience: int = 0,
    tolerance: float = 0.01,
    train_steps: int = 0,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    warmup_ratio: float = 0.06,
    run_name: str = "default",
    report_to: str = "none",
    checkpoints: bool = False,
    callbacks: tuple[TrainerCallback, ...] = (),
) -> Trainer:
    """Build a Trainer for the architecture, adding early stopping if requested."""
    spec = _build_spec(
        model,
        tokenizer=tokenizer,
        data=data,
        arch=arch,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        output_dir=output_dir,
        eval_steps=eval_steps,
        train_steps=train_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        run_name=run_name,
        report_to=report_to,
        checkpoints=checkpoints,
        callbacks=callbacks,
    )

    if arch == "encoder-decoder":
        trainer = _seq2seq_trainer(spec)
    elif arch == "decoder":
        trainer = _decoder_trainer(spec)
    else:
        trainer = _encoder_trainer(spec)

    _maybe_add_early_stopping(
        trainer,
        eval_steps=eval_steps,
        patience=patience,
        tolerance=tolerance,
    )

    return trainer


def _maybe_add_early_stopping(
    trainer: Trainer,
    *,
    eval_steps: int,
    patience: int,
    tolerance: float,
) -> None:
    """Attach early stopping when evaluation and patience are enabled."""
    if patience > 0 and eval_steps == 0:
        logger.warning("not using early stopping because not running evaluation")
    elif eval_steps > 0 and patience > 0:
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


def _allowed_next(
    completions: tuple[tuple[int, ...], ...],
    fallback: int,
    tail: tuple[int, ...],
) -> list[int]:
    """Return legal continuations of `tail`, falling back to `fallback`."""
    out = {
        completion[len(tail)]
        for completion in completions
        if len(completion) > len(tail) and completion[: len(tail)] == tail
    }

    return sorted(out) if out else [fallback]


def _label_completions(
    data: DataSplits,
    *,
    eos_token_id: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Collect distinct completion token sequences from the eval stream."""
    seen: list[tuple[int, ...]] = []
    for row in data.eval:
        labels = cast("list[int]", row["labels"])
        completion = tuple(token for token in labels if token != SENTINEL_TOKEN)
        if not completion or completion == (eos_token_id,):
            continue

        if completion not in seen:
            seen.append(completion)

    return tuple(seen)


def _constrain_eval_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    completions: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    """Force each row's argmax to its most likely legal completion."""
    if not completions:
        return logits

    masked = logits.clone()
    for b in range(masked.shape[0]):
        pos = torch.where(labels[b] != SENTINEL_TOKEN)[0]
        if len(pos) == 0:
            continue

        force_at = pos - 1
        if bool((force_at < 0).any()):
            continue

        cands = [c for c in completions if len(c) == len(pos)]
        if not cands:
            continue

        cand_ids = torch.tensor(cands, dtype=torch.long, device=masked.device)

        scores = masked[b, force_at].gather(1, cand_ids.t()).sum(0)
        winner = cands[int(torch.argmax(scores))]
        forced = torch.tensor(winner, dtype=torch.long, device=masked.device)

        masked[b, force_at, :] = float("-inf")
        masked[b, force_at, forced] = 0.0

    return masked
