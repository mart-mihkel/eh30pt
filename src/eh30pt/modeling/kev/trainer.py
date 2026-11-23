"""Hugging Face Trainer integration for Kev's pointer-scoring objective."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast, override

import torch
from datasets import Dataset
from torch.nn import functional
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from eh30pt.modeling.metrics import compute_classification_metrics
from eh30pt.modeling.training.callbacks import use_rich_logging

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from transformers import EvalPrediction, TrainerCallback

    from eh30pt.modeling.kev.model import KevModel
    from eh30pt.modeling.kev.prompt import EncodedSample


def _masked(logits: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """Exclude padded options from the loss and predictions."""
    return logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)


def _collate(
    samples: Sequence[Mapping[str, Any]], pad_id: int
) -> dict[str, torch.Tensor]:
    """Pad encoded Kev rows into a batch accepted by the model."""
    rows = len(samples)
    length = max(len(sample["input_ids"]) for sample in samples)
    options = max(len(sample["option_index"]) for sample in samples)
    batch = {
        "input_ids": torch.full((rows, length), pad_id, dtype=torch.long),
        "attention_mask": torch.zeros((rows, length), dtype=torch.long),
        "position_ids": torch.zeros((rows, length), dtype=torch.long),
        "decide_index": torch.zeros(rows, dtype=torch.long),
        "option_index": torch.zeros((rows, options), dtype=torch.long),
        "option_mask": torch.zeros((rows, options), dtype=torch.bool),
        "labels": torch.zeros(rows, dtype=torch.long),
    }
    for index, sample in enumerate(samples):
        ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        row_options = torch.tensor(sample["option_index"], dtype=torch.long)
        batch["input_ids"][index, : len(ids)] = ids
        batch["attention_mask"][index, : len(ids)] = 1
        batch["position_ids"][index, : len(ids)] = torch.arange(len(ids))
        batch["decide_index"][index] = sample["decide_index"]
        batch["option_index"][index, : len(row_options)] = row_options
        batch["option_mask"][index, : len(row_options)] = True
        batch["labels"][index] = sample["labels"]

    return batch


def _dataset(samples: Sequence[EncodedSample]) -> Dataset:
    """Convert Kev rows to the map-style dataset expected by Trainer."""
    return Dataset.from_list(
        [
            {
                "input_ids": sample.encoded.input_ids,
                "decide_index": sample.encoded.decide_index,
                "option_index": sample.encoded.option_indices,
                "labels": sample.label,
            }
            for sample in samples
        ]
    )


def _compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
    """Compute accuracy and macro F1 for Trainer evaluation results."""
    logits = torch.as_tensor(prediction.predictions)
    labels = torch.as_tensor(prediction.label_ids)
    return compute_classification_metrics(
        labels.cpu().numpy(),
        logits.argmax(dim=-1).cpu().numpy(),
        classes=list(range(logits.shape[-1])),
    )


class KevTrainer(Trainer):
    """Trainer that computes cross-entropy over Kev's variable option sets."""

    on_step: Callable[[int, Mapping[str, float]], None] | None = None
    on_epoch: Callable[[int, Mapping[str, float], int], None] | None = None

    @override
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        labels = inputs.pop("labels")
        option_mask = inputs.pop("option_mask")
        logits = cast("KevModel", model)(**inputs)
        logits = _masked(logits, option_mask)
        loss = functional.cross_entropy(logits, labels)
        if return_outputs:
            return loss, {"logits": logits}

        return loss

    @override
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        super().log(logs, start_time=start_time)
        if self.on_step is not None and "loss" in logs:
            self.on_step(self.state.global_step, {"train_loss": logs["loss"]})
        if self.on_epoch is not None and "eval_loss" in logs:
            self.on_epoch(
                int(self.state.epoch or 0),
                {
                    key: value
                    for key, value in logs.items()
                    if key in {"train_loss", "eval_loss", "eval_accuracy", "eval_f1"}
                },
                self.state.global_step,
            )


def get_kev_trainer(
    model: KevModel,
    *,
    train: Sequence[EncodedSample],
    evaluation: Sequence[EncodedSample],
    batch_size: int,
    epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    patience: int,
    tolerance: float,
    output_dir: Path | str,
    seed: int,
    callbacks: tuple[TrainerCallback, ...] = (),
    on_step: Callable[[int, Mapping[str, float]], None] | None = None,
    on_epoch: Callable[[int, Mapping[str, float], int], None] | None = None,
) -> KevTrainer:
    """Configure and run the shared Transformers training lifecycle."""
    device = next(model.parameters()).device
    has_evaluation = bool(evaluation)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=math.ceil(
            math.ceil(epochs * len(train) / batch_size) * warmup_ratio
        ),
        eval_strategy="epoch" if has_evaluation else "no",
        save_strategy="epoch" if has_evaluation else "no",
        load_best_model_at_end=has_evaluation,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        use_cpu=device.type == "cpu",
        bf16=device.type == "cuda",
        seed=seed,
    )
    trainer = KevTrainer(
        model=model,
        args=arguments,
        train_dataset=_dataset(train),
        eval_dataset=_dataset(evaluation) if has_evaluation else None,
        data_collator=lambda samples: _collate(samples, model.pad_id),
        compute_metrics=_compute_metrics,
        callbacks=[
            *callbacks,
            *(
                (EarlyStoppingCallback(patience, tolerance),)
                if has_evaluation and patience
                else ()
            ),
        ],
    )
    trainer.on_step = on_step
    trainer.on_epoch = on_epoch
    use_rich_logging(trainer)
    return trainer
