"""QLoRA-tune a quantized pretrained model and run test evaluation."""

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
from eh30pt.modeling.lora import prepare_lora_model
from eh30pt.modeling.metrics import MetricsAccumulator
from eh30pt.modeling.tokenizer import cap_tokenizer_length, load_tokenizer
from eh30pt.tracking.context import RunContext, registry_name
from eh30pt.tracking.resume import build_run_name

if TYPE_CHECKING:
    from eh30pt.settings import (
        CommonSettings,
        LoRASettings,
        MlflowSettings,
        TrainingSettings,
    )
    from eh30pt.types import Method, QLoRaRunConfig


def main(
    *,
    common: CommonSettings,
    lora: LoRASettings,
    training: TrainingSettings,
    mlflow: MlflowSettings,
) -> None:
    """QLoRA-tune `model_path` on `dataset` and evaluate on the test split."""
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
        seed=common.seed,
        data_path=common.data_path,
        train_size=training.train_size,
    )

    sys_prompt = get_sys_prompt(common.dataset)
    model = prepare_lora_model(
        tokenizer,
        common.model,
        arch,
        common.dataset,
        lora=lora,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    method: Method = "qlora"
    if mlflow.run_name is None:
        mlflow.run_name = build_run_name(common.model, method)

    logger.info("total parameters %d", total)
    logger.info("trainable parameters %d", trainable)
    logger.info(
        "tracking '%s' of experiment '%s'",
        mlflow.run_name,
        mlflow.experiment,
    )

    config: QLoRaRunConfig = {
        "method": method,
        "dataset": common.dataset,
        "architecture": arch,
        "base_model": common.model,
        "lora_r": lora.rank,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "n_shot": common.n_shot,
        "eval_steps": training.eval_steps,
        "patience": training.patience,
        "tolerance": training.tolerance,
        "epochs": training.epochs,
        "train_size": training.train_size,
        "save_model": training.save_model,
        "batch_size": training.batch_size,
        "learning_rate": training.learning_rate,
        "warmup_ratio": training.warmup_ratio,
        "constrain": training.constrain,
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
