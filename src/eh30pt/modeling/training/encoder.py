"""Classification Trainer for encoder-only models."""

from typing import TYPE_CHECKING

from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from eh30pt.logging import logger
from eh30pt.modeling.training.callbacks import use_rich_logging
from eh30pt.modeling.training.spec import _no_epoch_cap

if TYPE_CHECKING:
    from eh30pt.modeling.training.spec import _Spec


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
    )

    trainer = Trainer(
        args=args,
        model=spec.model,
        data_collator=spec.collate_fn,
        eval_dataset=spec.eval_dataset,
        train_dataset=spec.train_dataset,
        compute_metrics=spec.metrics_fn,
        callbacks=spec.callbacks,
    )

    use_rich_logging(trainer)
    return trainer
