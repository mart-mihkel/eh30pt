"""SFT Trainer for causal models."""

from typing import TYPE_CHECKING, Any, cast, override

import torch
from datasets import Dataset
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer

from eh30pt.logging import logger
from eh30pt.modeling.training.callbacks import use_rich_logging
from eh30pt.modeling.training.spec import _no_epoch_cap

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel, Trainer

    from eh30pt.modeling.training.spec import _Spec


class _DecoderTrainer(SFTTrainer):
    @override
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        del ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, return_outputs=False)
            if prediction_loss_only:
                return loss, None, None

            labels = inputs.get("labels")
            logits_inputs = {
                key: value for key, value in inputs.items() if key != "labels"
            }

            logits = model(**logits_inputs).logits
            return loss, logits, labels


def _decoder_trainer(spec: _Spec) -> Trainer:
    """Build the SFT Trainer for causal models."""
    logger.debug("using SFT trainer")

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

    train_dataset = (
        Dataset.from_list(
            [
                dict(spec.train_dataset[index])
                for index in tqdm(
                    range(len(spec.train_dataset)), desc="Tokenizing train"
                )
            ]
        )
        if spec.train_dataset is not None
        else None
    )

    eval_dataset = Dataset.from_list(
        [
            dict(spec.eval_dataset[index])
            for index in tqdm(range(len(spec.eval_dataset)), desc="Tokenizing eval")
        ]
    )

    trainer = _DecoderTrainer(
        model=cast("PreTrainedModel | PeftModel", spec.model),
        args=args,
        data_collator=spec.collate_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=spec.tokenizer,
        compute_metrics=spec.metrics_fn,
        callbacks=spec.callbacks,
    )

    use_rich_logging(trainer)
    return trainer
