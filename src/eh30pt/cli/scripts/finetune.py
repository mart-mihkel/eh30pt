"""Fine-tune a pretrained model and run test evaluation."""

from typing import TYPE_CHECKING

from transformers import AutoConfig

from eh30pt.datasets.registry import (
    assert_classification,
    get_sys_prompt,
    load_data,
)
from eh30pt.logging import logger
from eh30pt.modeling.arch import infer_architecture
from eh30pt.modeling.collator import get_collator
from eh30pt.modeling.loader import load_model
from eh30pt.modeling.metrics import MetricsAccumulator
from eh30pt.modeling.tokenizer import cap_tokenizer_length, load_tokenizer
from eh30pt.tracking.context import RunContext, registry_name
from eh30pt.tracking.resume import build_run_name

if TYPE_CHECKING:
    from eh30pt.settings import CommonSettings, MlflowSettings, TrainingSettings
    from eh30pt.types import FineTuneRunConfig, Method


def main(
    common: CommonSettings,
    *,
    head_only: bool,
    training: TrainingSettings,
    mlflow: MlflowSettings,
) -> None:
    """Fine-tune `model_path` on `dataset` and evaluate on the test split."""
    assert_classification(common.dataset, "training needs labels")

    logger.info("loading config for '%s'", common.model)
    config = AutoConfig.from_pretrained(common.model)
    arch = infer_architecture(config, common.arch)

    tokenizer = load_tokenizer(common.model)
    cap_tokenizer_length(tokenizer, config)

    collate_fn = get_collator(tokenizer, arch)
    metrics_fn = MetricsAccumulator(tokenizer, arch)
    splits = load_data(
        tokenizer,
        common.dataset,
        arch,
        n_shot=common.n_shot,
        labeling_projects=common.labeling_projects,
    )

    sys_prompt = get_sys_prompt(common.dataset)
    model = load_model(
        tokenizer, common.model, common.dataset, arch, head_only=head_only
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    method: Method = "cls-head" if head_only else "fine-tune"
    if mlflow.run_name is None:
        mlflow.run_name = build_run_name(common.model, method)

    logger.info("total parameters %d", total)
    logger.info("trainable parameters %d", trainable)
    logger.info(
        "tracking '%s' of experiment '%s'",
        mlflow.run_name,
        mlflow.experiment,
    )

    config: FineTuneRunConfig = {
        "method": method,
        "dataset": common.dataset,
        "architecture": arch,
        "base_model": common.model,
        "head_only": head_only,
        "n_shot": common.n_shot,
        "eval_steps": training.eval_steps,
        "patience": training.patience,
        "tolerance": training.tolerance,
        "train_steps": training.train_steps,
        "batch_size": training.batch_size,
        "learning_rate": training.learning_rate,
        "warmup_ratio": training.warmup_ratio,
        "system_prompt": sys_prompt,
        "total_parameters": total,
        "trainable_parameters": trainable,
    }

    with RunContext(mlflow, config) as ctx:
        registered_name = registry_name(
            mlflow.experiment,
            common.dataset,
            arch,
            method,
        )

        ctx.log_prompt_version(common.dataset, arch, sys_prompt)
        ctx.train_and_evaluate(
            model=model,
            tokenizer=tokenizer,
            data=splits,
            arch=arch,
            collate_fn=collate_fn,
            metrics_fn=metrics_fn,
            registered_name=registered_name,
            training=training,
        )
