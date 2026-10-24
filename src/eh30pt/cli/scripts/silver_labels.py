"""Generate silver predictions for existing Label Studio tasks."""

from typing import TYPE_CHECKING

from eh30pt.datasets.registry import get_few_shots, get_sys_prompt
from eh30pt.labeling.registry import predict_silver_tasks
from eh30pt.labeling.studio import StudioClient, project_tasks
from eh30pt.logging import logger
from eh30pt.openai.complete import CompletionClient
from eh30pt.openai.stream import CompletionPredictor
from eh30pt.tracking.context import RunContext
from eh30pt.tracking.resume import build_run_name

if TYPE_CHECKING:
    from eh30pt.settings import LabelStudioSettings, MlflowSettings, ServerSettings
    from eh30pt.types import DatasetName, FewShotRunConfig, Method


async def main(
    *,
    system_prompt: str | None,
    server: ServerSettings,
    studio: LabelStudioSettings,
    dataset: DatasetName,
    n_shot: int,
    concurrency: int,
    project: int,
    mlflow: MlflowSettings,
    seed: int | None,
) -> None:
    """Attach model predictions to existing Label Studio tasks."""
    if system_prompt is None:
        logger.debug("using '%s' system prompt", dataset)
        system_prompt = get_sys_prompt(dataset)

    compc = CompletionClient(server, seed)
    examples = get_few_shots(dataset, n_shot)
    model = (await compc.model()).unwrap()

    method: Method = "silver-labels"
    if mlflow.run_name is None:
        mlflow.run_name = build_run_name(model, method, f"{n_shot}-shot")

    logger.info(
        "tracking '%s' of experiment '%s'",
        mlflow.run_name,
        mlflow.experiment,
    )

    config: FewShotRunConfig = {
        "method": method,
        "dataset": dataset,
        "architecture": "decoder",
        "base_model": model,
        "n_shot": n_shot,
        "batch_size": 1,
        "system_prompt": system_prompt,
    }

    conv = compc.build_conversation(system=system_prompt, examples=examples)
    predictor = CompletionPredictor(client=compc, conversation=conv)

    with RunContext(mlflow, config) as ctx, StudioClient(studio) as studioc:
        ctx.log_prompt_version(dataset, "decoder", system_prompt)
        tasks = project_tasks(studioc, project)
        counts = await predict_silver_tasks(
            dataset,
            tasks,
            predictor,
            studioc,
            model=model,
            concurrency=concurrency,
        )

        logger.info(
            "predicted %d/%d tasks (%d invalid)",
            counts.predicted,
            len(tasks),
            counts.invalid,
        )
