"""Label-constrained seq2seq Trainer for encoder-decoder models."""

from typing import TYPE_CHECKING, Any, cast

from transformers import (
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from eh30pt.logging import logger
from eh30pt.modeling.training.callbacks import use_rich_logging
from eh30pt.modeling.training.constraints import _allowed_next
from eh30pt.modeling.training.spec import _no_epoch_cap

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor
    from torch.nn import Module
    from torch.utils.data import IterableDataset as TorchIterableDataset
    from transformers import DataCollator, Trainer, TrainerCallback

    from eh30pt.modeling.training.spec import _Spec


class ConstrainedSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer constraining generation to legal label token sequences.

    Generation is masked through `PrefixConstrainedLogitsProcessor` to tokens
    continuing the decoded prefix toward a legal label. An empty
    `label_completions` leaves generation unconstrained.
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

        def allowed(_batch_id: int, sent: Tensor) -> list[int]:
            """Return legal continuations past the decoder start token."""
            tail: tuple[int, ...] = tuple(sent[1:].tolist())
            return _allowed_next(label_completions, pad_token_id, tail)

        self._gen_kwargs: dict[str, Any] = {}

        if label_completions:
            self._gen_kwargs["logits_processor"] = LogitsProcessorList(
                [PrefixConstrainedLogitsProcessor(allowed, num_beams=1)]
            )

    def prediction_step(
        self,
        model: Module,
        inputs: dict[str, Tensor],
        prediction_loss_only: bool,  # noqa: FBT001
        ignore_keys: list[str] | None = None,
        **gen_kwargs: Any,  # noqa: ANN401
    ) -> tuple[float | None, Tensor | None, Tensor | None]:
        """Generate with the label-constraining logits processor injected."""
        inputs = {k: v for k, v in inputs.items() if k != "truncated"}
        merged = {**self._gen_kwargs, **gen_kwargs}

        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
            **merged,
        )


def _seq2seq_trainer(spec: _Spec) -> Trainer:
    """Build the label-constrained seq2seq Trainer for encoder-decoder models."""
    logger.debug(
        "using %s seq2seq trainer",
        "constrained" if spec.constrain else "unconstrained",
    )

    args = Seq2SeqTrainingArguments(
        run_name=spec.run_name,
        report_to=spec.report_to,
        output_dir=spec.output_dir,
        save_strategy=spec.save_strategy,
        save_steps=spec.save_steps,
        save_total_limit=1,
        eval_strategy=spec.eval_strategy,
        eval_steps=spec.eval_steps,
        eval_on_start=spec.eval_strategy != "no",
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

    trainer = ConstrainedSeq2SeqTrainer(
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

    use_rich_logging(trainer)
    return trainer
