"""Label-constrained SFT Trainer for causal models."""

from typing import TYPE_CHECKING, cast

from trl import SFTConfig, SFTTrainer

from eh30pt.logging import logger
from eh30pt.modeling.training.callbacks import use_rich_logging
from eh30pt.modeling.training.constraints import (
    _constrain_eval_logits,
    _constrained_causal_loss,
)
from eh30pt.modeling.training.spec import _no_epoch_cap

if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets.iterable_dataset import IterableDataset as HFIterableDataset
    from peft import PeftModel
    from torch import Tensor
    from torch.nn import Module
    from torch.utils.data import IterableDataset as TorchIterableDataset
    from transformers import (
        DataCollator,
        PreTrainedModel,
        PreTrainedTokenizerFast,
        Trainer,
        TrainerCallback,
    )

    from eh30pt.modeling.training.spec import _Spec


class ConstrainedSFTTrainer(SFTTrainer):
    """
    SFTTrainer constraining eval predictions to legal label token sequences.

    Each row's argmax is forced to its most likely legal completion, scored
    by teacher-forced likelihood over same-length candidates.

    An empty `label_completions` disables the constraint, leaving the raw
    model argmax in place. A non-empty one also constrains training, by
    renormalizing the softmax over the legal next tokens at each label step.
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
            compute_loss_func=(
                _constrained_causal_loss(label_completions)
                if label_completions
                else None
            ),
        )

        self._completions = label_completions

    def prediction_step(
        self,
        model: Module,
        inputs: dict[str, Tensor],
        prediction_loss_only: bool,  # noqa: FBT001
        ignore_keys: list[str] | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Constrain eval logits to legal label continuations."""
        loss, logits, labels = super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )

        if isinstance(logits, tuple):
            logits = logits[0] if logits else None

        if prediction_loss_only or logits is None or labels is None:
            return loss, logits, labels

        if not self._completions:
            return loss, logits, labels

        return loss, _constrain_eval_logits(logits, labels, self._completions), labels


def _decoder_trainer(spec: _Spec) -> Trainer:
    """Build the label-constrained SFT Trainer for causal models."""
    logger.debug(
        "using %s SFT trainer",
        "constrained" if spec.constrain else "unconstrained",
    )

    args = SFTConfig(
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
        dataset_kwargs={"skip_prepare_dataset": True},
        packing=False,
        loss_type="nll",
    )

    trainer = ConstrainedSFTTrainer(
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

    use_rich_logging(trainer)
    return trainer
