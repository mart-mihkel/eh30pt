"""Seq2seq Trainer for encoder-decoder models."""

from typing import TYPE_CHECKING, Any

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from eh30pt.logging import logger
from eh30pt.modeling.training.callbacks import use_rich_logging
from eh30pt.modeling.training.spec import _no_epoch_cap

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from transformers import Trainer

    from eh30pt.modeling.training.spec import _Spec


class _Seq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer that drops the streaming truncation flag before generation."""

    def prediction_step(
        self,
        model: Module,
        inputs: dict[str, Tensor],
        prediction_loss_only: bool,  # noqa: FBT001
        ignore_keys: list[str] | None = None,
        **gen_kwargs: Any,  # noqa: ANN401
    ) -> tuple[float | None, Tensor | None, Tensor | None]:
        """Generate without passing the extra streaming flag to the model."""
        inputs = {key: value for key, value in inputs.items() if key != "truncated"}

        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )


def _seq2seq_trainer(spec: _Spec) -> Trainer:
    """Build the seq2seq Trainer for encoder-decoder models."""
    logger.debug("using seq2seq trainer")

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

    trainer = _Seq2SeqTrainer(
        model=spec.model,
        args=args,
        data_collator=spec.collate_fn,
        train_dataset=spec.train_dataset,
        eval_dataset=spec.eval_dataset,
        compute_metrics=spec.metrics_fn,
        callbacks=spec.callbacks,
    )

    use_rich_logging(trainer)
    return trainer
