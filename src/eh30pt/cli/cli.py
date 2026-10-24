"""Command line interface."""

import asyncio
from typing import TYPE_CHECKING

import rich_click as click

from eh30pt.cli.options.common import (
    arch_option,
    dataset_option,
    inferred_dataset_option,
    lang_extract_n_shot_option,
    log_level_option,
    model_option,
    n_shot_option,
    output_option,
    seed_option,
    system_prompt_option,
)
from eh30pt.cli.options.labeling import (
    data_path_option,
    label_studio_url_option,
    project_id_option,
    project_ids_option,
    project_title_option,
)
from eh30pt.cli.options.lora import (
    lora_alpha_option,
    lora_dropout_option,
    lora_rank_option,
)
from eh30pt.cli.options.mlflow import (
    experiment_option,
    export_run_option,
    run_id_option,
    run_name_option,
    tracking_uri_option,
)
from eh30pt.cli.options.server import (
    completion_dataset_option,
    concurrency_option,
    log_interval_option,
    prompt_argument,
    provider_option,
    shard_size_option,
)
from eh30pt.cli.options.slurm import (
    dry_run_option,
    inner_command_argument,
    slurm_cpus_option,
    slurm_gres_option,
    slurm_job_name_option,
    slurm_mem_option,
    slurm_output_option,
    slurm_partition_option,
    slurm_time_option,
)
from eh30pt.cli.options.training import (
    batch_size_option,
    epochs_option,
    eval_steps_option,
    head_only_option,
    learning_rate_option,
    num_virtual_option,
    patience_option,
    prefix_init_option,
    save_model_option,
    sequence_classification_option,
    tolerance_option,
    train_size_option,
    warmup_ratio_option,
)
from eh30pt.logging import logger, setup_logging
from eh30pt.settings import (
    CommonSettings,
    LoRASettings,
    MlflowSettings,
    ServerSettings,
    StreamSettings,
    TrainingSettings,
)

if TYPE_CHECKING:
    from pathlib import Path

    from eh30pt.types import (
        Architecture,
        DatasetName,
        EvalSteps,
        LogLevel,
        PrefixInit,
        Provider,
    )


@click.group(
    "ehpt",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Est Health 30 prompt-tuning study command line interface",
    no_args_is_help=True,
)
@click.rich_config(
    click.RichHelpConfiguration(
        theme="cargo-slim",
        command_groups={
            "ehpt": [
                {
                    "name": "Hugging Face",
                    "commands": ["fine-tune", "prompt-tune", "qlora", "few-shot"],
                },
                {
                    "name": "OpenAI",
                    "commands": [
                        "completion",
                        "evaluate",
                        "generate",
                        "langextract",
                        "silver-labels",
                    ],
                },
                {
                    "name": "Label-Studio",
                    "commands": ["create-task", "export-labels"],
                },
                {"name": "Meta", "commands": ["slurm"]},
            ]
        },
        option_groups={
            "ehpt *": [
                {
                    "name": "Model",
                    "options": ["--model", "--arch", "--sequence-classification"],
                },
                {
                    "name": "Data",
                    "options": [
                        "--dataset",
                        "--n-shot",
                        "--project",
                        "--data-path",
                    ],
                },
                {
                    "name": "Training",
                    "options": [
                        "--epochs",
                        "--batch-size",
                        "--learning-rate",
                        "--warmup-ratio",
                        "--train-size",
                        "--eval-steps",
                        "--patience",
                        "--tolerance",
                        "--head-only",
                        "--lora-rank",
                        "--lora-alpha",
                        "--lora-dropout",
                        "--save-model",
                    ],
                },
                {
                    "name": "Prompt",
                    "options": [
                        "--system",
                        "--prefix-init",
                        "--num-virtual",
                    ],
                },
                {
                    "name": "Server",
                    "options": [
                        "--provider",
                        "--concurrency",
                        "--shard-size",
                        "--log-interval",
                    ],
                },
                {
                    "name": "SLURM",
                    "options": [
                        "--job-name",
                        "--partition",
                        "--gres",
                        "--cpus",
                        "--mem",
                        "--time",
                        "--dry-run",
                    ],
                },
                {"name": "Output", "options": ["--output"]},
                {
                    "name": "MLflow",
                    "options": [
                        "--mlflow-tracking-uri",
                        "--mlflow-experiment",
                        "--mlflow-run-name",
                        "--mlflow-run-id",
                    ],
                },
                {
                    "name": "Label Studio",
                    "options": ["--label-studio-url", "--title"],
                },
                {"name": "Run", "options": ["--log-level", "--seed"]},
            ]
        },
    )
)
def app() -> None:
    """Set up logging before running a command."""
    setup_logging()


def set_seed(seed: int | None) -> None:
    """Fix randomness."""
    import random

    import numpy as np
    import torch

    if seed is None:
        logger.warning("randomness is not fixed")
        return

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.mps.is_available():
        torch.mps.manual_seed(seed)


def log_accelerator() -> None:
    """Log torch acceleator."""
    import torch

    if torch.cuda.is_available():
        logger.info("cuda available")
    elif torch.mps.is_available():
        logger.info("metal available")
    else:
        logger.warning("no acceleator available")


@app.command("fine-tune", help="Fine-tune and run test evaluation")
@model_option
@dataset_option
@head_only_option
@sequence_classification_option
@n_shot_option
@system_prompt_option
@epochs_option
@train_size_option
@save_model_option
@eval_steps_option
@batch_size_option
@learning_rate_option(5e-5)
@warmup_ratio_option(0)
@patience_option
@tolerance_option
@arch_option
@data_path_option
@experiment_option
@run_name_option
@run_id_option
@tracking_uri_option
@log_level_option
@seed_option
def fine_tune(
    *,
    model: str,
    dataset: DatasetName,
    head_only: bool,
    sequence_classification: bool,
    n_shot: int,
    system_prompt: str | None,
    epochs: float,
    train_size: int | None,
    save_model: bool,
    eval_steps: EvalSteps,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    patience: int,
    tolerance: float,
    arch: Architecture | None,
    data_path: Path | None,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
    mlflow_tracking_uri: str | None,
    log_level: LogLevel,
    seed: int | None,
) -> None:
    """Fine-tune and run test evaluation."""
    from eh30pt.cli.scripts.finetune import main

    logger.setLevel(log_level.upper())
    set_seed(seed)
    log_accelerator()

    common = CommonSettings(
        model=model,
        dataset=dataset,
        arch=arch,
        sequence_classification=sequence_classification,
        n_shot=n_shot,
        system_prompt=system_prompt,
        data_path=data_path,
        seed=seed,
    )

    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri or MlflowSettings().tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    training = TrainingSettings(
        eval_steps=eval_steps,
        patience=patience,
        tolerance=tolerance,
        epochs=epochs,
        train_size=train_size,
        save_model=save_model,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
    )

    main(
        common=common,
        head_only=head_only,
        training=training,
        mlflow=mlflow,
    )


@app.command("prompt-tune", help="Prompt-tune and run test evaluation")
@model_option
@dataset_option
@prefix_init_option
@num_virtual_option
@sequence_classification_option
@n_shot_option
@system_prompt_option
@epochs_option
@train_size_option
@save_model_option
@eval_steps_option
@batch_size_option
@learning_rate_option(1e-2)
@warmup_ratio_option(0)
@patience_option
@tolerance_option
@arch_option
@data_path_option
@experiment_option
@run_name_option
@run_id_option
@tracking_uri_option
@log_level_option
@seed_option
def prompt_tune(
    *,
    model: str,
    dataset: DatasetName,
    prefix_init: PrefixInit,
    num_virtual: int | None,
    sequence_classification: bool,
    n_shot: int,
    system_prompt: str | None,
    epochs: float,
    train_size: int | None,
    save_model: bool,
    eval_steps: EvalSteps,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    patience: int,
    tolerance: float,
    arch: Architecture | None,
    data_path: Path | None,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
    mlflow_tracking_uri: str | None,
    log_level: LogLevel,
    seed: int | None,
) -> None:
    """Prompt-tune and run test evaluation."""
    from eh30pt.cli.scripts.prompttune import main

    logger.setLevel(log_level.upper())
    set_seed(seed)
    log_accelerator()

    common = CommonSettings(
        model=model,
        dataset=dataset,
        arch=arch,
        sequence_classification=sequence_classification,
        n_shot=n_shot,
        system_prompt=system_prompt,
        data_path=data_path,
        seed=seed,
    )

    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri or MlflowSettings().tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    training = TrainingSettings(
        eval_steps=eval_steps,
        patience=patience,
        tolerance=tolerance,
        epochs=epochs,
        train_size=train_size,
        save_model=save_model,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
    )

    main(
        common=common,
        prefix_init=prefix_init,
        num_virtual=num_virtual,
        training=training,
        mlflow=mlflow,
    )


@app.command("qlora", help="QLoRA-tune and run test evaluation")
@model_option
@dataset_option
@lora_rank_option
@lora_alpha_option
@lora_dropout_option
@sequence_classification_option
@n_shot_option
@system_prompt_option
@epochs_option
@train_size_option
@save_model_option
@eval_steps_option
@batch_size_option
@learning_rate_option(2e-4)
@warmup_ratio_option(0)
@patience_option
@tolerance_option
@arch_option
@data_path_option
@experiment_option
@run_name_option
@run_id_option
@tracking_uri_option
@log_level_option
@seed_option
def qlora(
    *,
    model: str,
    dataset: DatasetName,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    sequence_classification: bool,
    n_shot: int,
    system_prompt: str | None,
    epochs: float,
    train_size: int | None,
    save_model: bool,
    eval_steps: EvalSteps,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    patience: int,
    tolerance: float,
    arch: Architecture | None,
    data_path: Path | None,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
    mlflow_tracking_uri: str | None,
    log_level: LogLevel,
    seed: int | None,
) -> None:
    """QLoRA-tune and run test evaluation."""
    from eh30pt.cli.scripts.qlora import main

    logger.setLevel(log_level.upper())
    set_seed(seed)
    log_accelerator()

    common = CommonSettings(
        model=model,
        dataset=dataset,
        arch=arch,
        sequence_classification=sequence_classification,
        n_shot=n_shot,
        system_prompt=system_prompt,
        data_path=data_path,
        seed=seed,
    )

    lora = LoRASettings(
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )

    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri or MlflowSettings().tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    training = TrainingSettings(
        eval_steps=eval_steps,
        patience=patience,
        tolerance=tolerance,
        epochs=epochs,
        train_size=train_size,
        save_model=save_model,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
    )

    main(
        common=common,
        lora=lora,
        training=training,
        mlflow=mlflow,
    )


@app.command("few-shot", help="Run test evaluation with few-shot learning")
@model_option
@dataset_option
@sequence_classification_option
@n_shot_option
@system_prompt_option
@batch_size_option
@arch_option
@data_path_option
@experiment_option
@run_name_option
@run_id_option
@tracking_uri_option
@log_level_option
@seed_option
def few_shot(
    *,
    model: str,
    dataset: DatasetName,
    sequence_classification: bool,
    n_shot: int,
    system_prompt: str | None,
    batch_size: int,
    arch: Architecture | None,
    data_path: Path | None,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
    mlflow_tracking_uri: str | None,
    log_level: LogLevel,
    seed: int | None,
) -> None:
    """Run test evaluation with few-shot learning."""
    from eh30pt.cli.scripts.fewshot import main

    logger.setLevel(log_level.upper())
    set_seed(seed)
    log_accelerator()

    common = CommonSettings(
        model=model,
        dataset=dataset,
        arch=arch,
        sequence_classification=sequence_classification,
        n_shot=n_shot,
        system_prompt=system_prompt,
        data_path=data_path,
        seed=seed,
    )

    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri or MlflowSettings().tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    main(common=common, batch_size=batch_size, mlflow=mlflow)


@app.command(
    "slurm",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    help="Submit a training command as a SLURM job",
)
@click.pass_context
@inner_command_argument
@slurm_job_name_option
@slurm_output_option
@slurm_partition_option
@slurm_gres_option
@slurm_cpus_option
@slurm_mem_option
@slurm_time_option
@dry_run_option
@log_level_option
def slurm(
    ctx: click.Context,
    command: str,
    *,
    job_name: str | None,
    output: str,
    partition: str | None,
    gres: str | None,
    cpus: int,
    mem: str,
    time: str,
    dry_run: bool,
    log_level: LogLevel,
) -> None:
    """
    Submit a training command as a SLURM job.

    Append the training command line after `--`.
    """
    from eh30pt.cli.scripts.slurm import main

    logger.setLevel(log_level.upper())

    if command not in ("fine-tune", "prompt-tune", "qlora", "few-shot"):
        msg = f"cannot submit '{command}', expected a training command"
        raise click.BadParameter(msg)

    main(
        command,
        list(ctx.args),
        job_name=job_name or command,
        output=output,
        partition=partition,
        gres=gres,
        cpus=cpus,
        mem=mem,
        time=time,
        dry_run=dry_run,
    )


@app.command("completion", help="Query a server and print the completion")
@prompt_argument
@system_prompt_option
@provider_option
@completion_dataset_option
@log_level_option
@seed_option
def completion(
    *,
    user_prompt: str,
    system_prompt: str | None,
    provider: Provider,
    dataset: DatasetName | None,
    log_level: LogLevel,
    seed: int | None,
) -> None:
    """Query a server and print the completion."""
    from eh30pt.cli.scripts.completion import main
    from eh30pt.datasets.registry import get_response_schema, get_sys_prompt

    logger.setLevel(log_level.upper())
    if seed is None:
        logger.warning("randomness is not fixed")

    if system_prompt is None and dataset is not None:
        logger.debug("using '%s' system prompt", dataset)
        system_prompt = get_sys_prompt(dataset)

    if system_prompt is None:
        msg = "system prompt is required when --dataset is not given"
        raise click.BadParameter(msg, param_hint="--system")

    fmt = get_response_schema(dataset) if dataset is not None else None
    server = ServerSettings(provider=provider, format=fmt)

    asyncio.run(
        main(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            server=server,
            seed=seed,
        )
    )


@app.command("evaluate", help="Run few-shot test evaluation on a server")
@dataset_option
@provider_option
@n_shot_option
@system_prompt_option
@concurrency_option
@run_name_option
@run_id_option
@experiment_option
@tracking_uri_option
@log_level_option
@log_interval_option
@seed_option
def evaluate(
    *,
    dataset: DatasetName,
    provider: Provider,
    n_shot: int,
    system_prompt: str | None,
    concurrency: int,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
    mlflow_experiment: str,
    mlflow_tracking_uri: str | None,
    log_level: LogLevel,
    log_interval: int,
    seed: int | None,
) -> None:
    """Run few-shot test evaluation on a server."""
    from eh30pt.cli.scripts.evaluate import main
    from eh30pt.datasets.registry import get_response_schema

    logger.setLevel(log_level.upper())
    if seed is None:
        logger.warning("randomness is not fixed")

    fmt = get_response_schema(dataset)
    server = ServerSettings(provider=provider, format=fmt)

    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri or MlflowSettings().tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    stream = StreamSettings(log_interval=log_interval, concurrency=concurrency)

    asyncio.run(
        main(
            system_prompt=system_prompt,
            server=server,
            dataset=dataset,
            n_shot=n_shot,
            stream=stream,
            mlflow=mlflow,
            seed=seed,
        ),
    )


@app.command("generate", help="Generate predictions on a server")
@dataset_option
@provider_option
@n_shot_option
@system_prompt_option
@concurrency_option
@run_name_option
@run_id_option
@experiment_option
@tracking_uri_option
@log_level_option
@shard_size_option
@seed_option
def generate(
    *,
    dataset: DatasetName,
    provider: Provider,
    n_shot: int,
    system_prompt: str | None,
    concurrency: int,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
    mlflow_experiment: str,
    mlflow_tracking_uri: str | None,
    log_level: LogLevel,
    shard_size: int,
    seed: int | None,
) -> None:
    """Generate predictions on a server and log them as an MLflow artifact."""
    from eh30pt.cli.scripts.generate import main
    from eh30pt.datasets.registry import get_response_schema

    logger.setLevel(log_level.upper())
    if seed is None:
        logger.warning("randomness is not fixed")

    fmt = get_response_schema(dataset)
    server = ServerSettings(provider=provider, format=fmt)

    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri or MlflowSettings().tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    stream = StreamSettings(shard_size=shard_size, concurrency=concurrency)

    asyncio.run(
        main(
            system_prompt=system_prompt,
            server=server,
            dataset=dataset,
            n_shot=n_shot,
            stream=stream,
            mlflow=mlflow,
            seed=seed,
        ),
    )


@app.command("silver-labels", help="Generate model predictions for Label Studio tasks")
@dataset_option
@provider_option
@n_shot_option
@system_prompt_option
@concurrency_option
@project_id_option
@label_studio_url_option
@run_name_option
@run_id_option
@experiment_option
@tracking_uri_option
@log_level_option
@seed_option
def silver_labels(
    *,
    dataset: DatasetName,
    provider: Provider,
    n_shot: int,
    system_prompt: str | None,
    concurrency: int,
    project: int,
    label_studio_url: str,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
    mlflow_experiment: str,
    mlflow_tracking_uri: str | None,
    log_level: LogLevel,
    seed: int | None,
) -> None:
    """Attach model predictions to existing Label Studio tasks."""
    from eh30pt.cli.scripts.silver_labels import main
    from eh30pt.datasets.registry import get_response_schema
    from eh30pt.settings import LabelStudioSettings

    logger.setLevel(log_level.upper())
    if seed is None:
        logger.warning("randomness is not fixed")

    fmt = get_response_schema(dataset)
    studio = LabelStudioSettings(url=label_studio_url)
    server = ServerSettings(provider=provider, format=fmt)
    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri or MlflowSettings().tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    asyncio.run(
        main(
            system_prompt=system_prompt,
            server=server,
            studio=studio,
            dataset=dataset,
            project=project,
            n_shot=n_shot,
            concurrency=concurrency,
            mlflow=mlflow,
            seed=seed,
        ),
    )


@app.command("langextract", help="Generate predictions with langextract on a server")
@dataset_option
@provider_option
@lang_extract_n_shot_option
@system_prompt_option
@run_name_option
@run_id_option
@experiment_option
@tracking_uri_option
@log_level_option
@shard_size_option
@concurrency_option
@seed_option
def langextract(
    *,
    dataset: DatasetName,
    provider: Provider,
    n_shot: int,
    system_prompt: str | None,
    mlflow_run_name: str | None,
    mlflow_run_id: str | None,
    mlflow_experiment: str,
    mlflow_tracking_uri: str | None,
    log_level: LogLevel,
    shard_size: int,
    concurrency: int,
    seed: int | None,
) -> None:
    """Generate predictions with langextract and log them as an artifact."""
    from eh30pt.cli.scripts.langextract import main

    logger.setLevel(log_level.upper())
    if seed is None:
        logger.warning("randomness is not fixed")

    server = ServerSettings(provider=provider)
    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri or MlflowSettings().tracking_uri,
        experiment=mlflow_experiment,
        run_name=mlflow_run_name,
        run_id=mlflow_run_id,
    )

    stream = StreamSettings(shard_size=shard_size, concurrency=concurrency)

    asyncio.run(
        main(
            system_prompt=system_prompt,
            server=server,
            dataset=dataset,
            n_shot=n_shot,
            stream=stream,
            mlflow=mlflow,
            seed=seed,
        ),
    )


@app.command("create-task", help="Create a Label Studio project from prediction shards")
@export_run_option
@inferred_dataset_option
@project_title_option
@label_studio_url_option
@tracking_uri_option
@log_level_option
def create_task(
    *,
    mlflow_run_id: str,
    dataset: DatasetName | None,
    title: str | None,
    label_studio_url: str,
    mlflow_tracking_uri: str | None,
    log_level: LogLevel,
) -> None:
    """Create a Label Studio project and import the run's tasks into it."""
    from eh30pt.cli.scripts.create_task import main
    from eh30pt.settings import LabelStudioSettings

    logger.setLevel(log_level.upper())

    studio = LabelStudioSettings(url=label_studio_url)
    mlflow = MlflowSettings(
        tracking_uri=mlflow_tracking_uri or MlflowSettings().tracking_uri,
        run_id=mlflow_run_id,
    )

    project = main(
        settings=studio,
        mlflow_settings=mlflow,
        dataset=dataset,
        title=title,
    )

    logger.info(
        "label studio project %d is ready at '%s'", project.id, label_studio_url
    )


@app.command(
    "export-labels",
    help="Export finished annotations from Label Studio projects",
)
@project_ids_option
@dataset_option
@output_option
@label_studio_url_option
@log_level_option
def export_labels(
    *,
    labeling_projects: tuple[int, ...],
    dataset: DatasetName,
    output: Path | None,
    label_studio_url: str,
    log_level: LogLevel,
) -> None:
    """Read finished annotations into one Parquet frame."""
    from eh30pt.cli.scripts.export_labels import main
    from eh30pt.settings import LabelStudioSettings

    logger.setLevel(log_level.upper())

    studio = LabelStudioSettings(url=label_studio_url)
    main(
        settings=studio,
        projects=list(labeling_projects),
        dataset=dataset,
        output=output,
    )


if __name__ == "__main__":
    app()
