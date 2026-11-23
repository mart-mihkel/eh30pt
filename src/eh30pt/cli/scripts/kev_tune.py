"""Train a Kev-style decision model on a labelled dataset."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from eh30pt.datasets.registry import (
    assert_classification,
    get_few_shots,
    get_labels,
    get_sys_prompt,
    load_data,
)
from eh30pt.logging import logger
from eh30pt.modeling.kev import (
    calibrate_kev,
    encode_records,
    evaluate_kev,
    load_kev_model,
    records_from_split,
    resolve_device,
)
from eh30pt.modeling.kev.trainer import get_kev_trainer
from eh30pt.modeling.tokenizer import load_tokenizer
from eh30pt.tracking.context import RunContext
from eh30pt.tracking.resume import build_run_name

if TYPE_CHECKING:
    from eh30pt.settings import (
        CommonSettings,
        KevSettings,
        LoRASettings,
        MlflowSettings,
        TrainingSettings,
    )
    from eh30pt.types import KevRunConfig, Method


def _base_revision(model: str) -> str | None:
    """Pin the base commit so conversion can reproduce the training backbone."""
    if Path(model).expanduser().is_dir():
        return None

    try:
        from huggingface_hub import model_info

        return model_info(model).sha
    except Exception:  # noqa: BLE001
        logger.debug("could not resolve the revision of '%s'", model)
        return None


def main(
    *,
    common: CommonSettings,
    kev: KevSettings,
    lora: LoRASettings,
    training: TrainingSettings,
    mlflow: MlflowSettings,
) -> None:
    """Train a decision model on `dataset` and evaluate its test split."""
    assert_classification(common.dataset, "decision training needs labels")

    labels = get_labels(common.dataset)
    sys_prompt = common.system_prompt or get_sys_prompt(common.dataset, "decoder")
    tokenizer = load_tokenizer(common.model)
    data = load_data(
        tokenizer,
        common.dataset,
        "decoder",
        n_shot=common.n_shot,
        seed=common.seed,
        data_path=common.data_path,
        train_size=training.train_size,
    )
    examples = get_few_shots(common.dataset, common.n_shot)

    train_records = records_from_split(
        data.train,
        instructions=sys_prompt,
        labels=labels,
        examples=examples,
    )
    eval_records = records_from_split(
        data.eval,
        instructions=sys_prompt,
        labels=labels,
        examples=examples,
    )
    test_records = records_from_split(
        data.test,
        instructions=sys_prompt,
        labels=labels,
        examples=examples,
    )
    if not train_records:
        msg = "no labelled decision records in the train split"
        raise ValueError(msg)

    train = encode_records(
        tokenizer,
        train_records,
        max_state=kev.max_state,
        max_branch=kev.max_branch,
    )
    evaluation = encode_records(
        tokenizer,
        eval_records,
        max_state=kev.max_state,
        max_branch=kev.max_branch,
    )
    test = encode_records(
        tokenizer,
        test_records,
        max_state=kev.max_state,
        max_branch=kev.max_branch,
    )
    if not train:
        msg = "all training records exceed the decision token budget"
        raise ValueError(msg)

    logger.info(
        "decision records: %d train, %d eval, %d test",
        len(train),
        len(evaluation),
        len(test),
    )

    device = resolve_device(kev.device)
    model = load_kev_model(
        common.model,
        tokenizer,
        device,
        lora=lora,
        head_dim=kev.head_dim,
        init_from=kev.init_from,
    )

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.trainable_parameters())
    logger.info("total parameters %d", total)
    logger.info("trainable parameters %d", trainable)

    method: Method = "kev-tune"
    if mlflow.run_name is None:
        mlflow.run_name = build_run_name(common.model, method, common.dataset)

    config: KevRunConfig = {
        "method": method,
        "dataset": common.dataset,
        "architecture": "decoder",
        "base_model": common.model,
        "n_shot": common.n_shot,
        "eval_steps": 0,
        "patience": training.patience,
        "tolerance": training.tolerance,
        "epochs": training.epochs,
        "train_size": training.train_size,
        "save_model": training.save_model,
        "batch_size": training.batch_size,
        "learning_rate": training.learning_rate,
        "warmup_ratio": training.warmup_ratio,
        "head_dim": kev.head_dim,
        "lora_rank": lora.rank,
        "init_from": kev.init_from,
        "max_state": kev.max_state,
        "max_branch": kev.max_branch,
        "system_prompt": sys_prompt,
        "total_parameters": total,
        "trainable_parameters": trainable,
    }

    logger.info(
        "tracking '%s' of experiment '%s'",
        mlflow.run_name,
        mlflow.experiment,
    )

    with RunContext(mlflow, config) as ctx:
        ctx.log_prompt_version(common.dataset, "decoder", sys_prompt)

        trainer = get_kev_trainer(
            model,
            train=train,
            evaluation=evaluation,
            batch_size=training.batch_size,
            epochs=training.epochs,
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            warmup_ratio=training.warmup_ratio,
            patience=training.patience,
            tolerance=training.tolerance,
            seed=common.seed or 0,
            output_dir=ctx.dir / "trainer-checkpoints",
            on_step=lambda step, values: ctx.log_metrics(values, step=step),
            on_epoch=lambda _epoch, values, step: ctx.log_metrics(values, step=step),
        )
        trainer.train()
        temperature = (
            calibrate_kev(model, evaluation, batch_size=training.batch_size)
            if evaluation
            else 1.0
        )
        steps = trainer.state.global_step

        test_metrics = evaluate_kev(
            model,
            test,
            batch_size=training.batch_size,
            temperature=temperature,
        )
        metrics = {
            "test_loss": test_metrics.loss,
            "test_accuracy": test_metrics.accuracy,
            "test_f1": test_metrics.f1,
            "temperature": temperature,
        }
        ctx.log_metrics(metrics, step=steps)
        ctx.log_params({"temperature": f"{temperature:.6f}"})

        if training.save_model:
            output_dir = ctx.dir / "checkpoint"
            model.save_checkpoint(
                tokenizer,
                output_dir,
                base_model=common.model,
                base_revision=_base_revision(common.model),
                lora_rank=lora.rank,
                temperature=temperature,
                training_args={
                    "base": common.model,
                    "dataset": common.dataset,
                    "epochs": training.epochs,
                    "batch_size": training.batch_size,
                    "learning_rate": training.learning_rate,
                    "weight_decay": training.weight_decay,
                    "lora": lora.rank,
                    "lora_alpha": lora.alpha,
                    "lora_dropout": lora.dropout,
                    "init_from": kev.init_from,
                    "max_state": kev.max_state,
                    "max_branch": kev.max_branch,
                    "n_shot": common.n_shot,
                    "seed": common.seed,
                    "temperature": temperature,
                },
            )
            ctx.log_artifacts(str(output_dir), "kev-model")

        report_path = ctx.dir / "kev_metrics.json"
        report_path.write_text(
            json.dumps(
                {
                    "steps": steps,
                    "temperature": temperature,
                    "history": trainer.state.log_history,
                    "test": asdict(test_metrics),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        ctx.log_artifact(report_path)

    logger.info(
        "test accuracy %.4f, macro f1 %.4f, loss %.4f",
        test_metrics.accuracy,
        test_metrics.f1,
        test_metrics.loss,
    )
