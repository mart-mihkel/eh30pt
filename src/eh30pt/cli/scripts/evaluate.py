"""Run few-shot test evaluation against a server."""

from typing import TYPE_CHECKING, cast

from tqdm import tqdm

from eh30pt.datasets.registry import (
    assert_classification,
    get_sys_prompt,
    load_data,
)
from eh30pt.logging import logger
from eh30pt.modeling.metrics import StreamingClassificationMetrics
from eh30pt.openai.complete import CompletionClient
from eh30pt.openai.stream import (
    CompletionPredictor,
    batched_concurrent,
    log_invalid,
    log_summary,
)
from eh30pt.tracking.context import RunContext
from eh30pt.tracking.resume import (
    build_run_name,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from transformers import PreTrainedTokenizerFast

    from eh30pt.settings import MlflowSettings, ServerSettings, StreamSettings
    from eh30pt.types import (
        DatasetName,
        FewShot,
        FewShotRunConfig,
        Method,
        Predictor,
    )


def _fold_eval_window(
    ctx: RunContext,
    accumulator: StreamingClassificationMetrics,
    window: list[FewShot],
    results: list[str | None],
    *,
    log_interval: int,
    n_seen: int,
    invalid: int,
    step_offset: int = 0,
) -> tuple[int, int]:
    """Fold one ordered window into the accumulator, returning (n_seen, invalid)."""
    for sample, prediction in zip(window, results, strict=True):
        n_seen += 1
        if prediction is None:
            invalid += 1
            continue

        accumulator.update(sample.label, prediction)
        if (n_seen + step_offset) % log_interval == 0:
            metrics = accumulator.finalize().items()
            metrics = {f"test_{key}": value for key, value in metrics}
            ctx.log_metrics(metrics, step=n_seen + step_offset)

    return n_seen, invalid


async def _stream_evaluate(
    *,
    predictor: Predictor[FewShot, str | None],
    samples: Iterator[FewShot],
    concurrency: int,
    log_interval: int,
    ctx: RunContext,
    accumulator: StreamingClassificationMetrics,
    bar: tqdm,
    step_offset: int = 0,
) -> tuple[int, int]:
    """Stream samples concurrently into the accumulator, returning (n_seen, invalid)."""
    n_seen = 0
    invalid = 0

    async for window, results in batched_concurrent(samples, predictor, concurrency):
        n_seen, invalid = _fold_eval_window(
            ctx,
            accumulator,
            window,
            results,
            log_interval=log_interval,
            n_seen=n_seen,
            invalid=invalid,
            step_offset=step_offset,
        )

        bar.update(len(window))

    return n_seen, invalid


async def main(
    *,
    system_prompt: str | None,
    server: ServerSettings,
    dataset: DatasetName,
    n_shot: int,
    stream: StreamSettings,
    mlflow: MlflowSettings,
    seed: int | None,
) -> None:
    """Run few-shot test evaluation on the server."""
    assert_classification(dataset, "metrics need labels")

    tokenizer = cast("PreTrainedTokenizerFast", None)
    data = load_data(
        tokenizer,
        dataset,
        "decoder",
        n_shot=n_shot,
    )

    if system_prompt is None:
        logger.debug("using '%s' system prompt", dataset)
        system_prompt = get_sys_prompt(dataset)

    client = CompletionClient(server, seed)
    model = (await client.model()).unwrap()

    method: Method = "few-shot"
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

    conv = client.build_conversation(system=system_prompt, examples=data.test.few_shots)
    predictor = CompletionPredictor(client=client, conversation=conv)

    with RunContext(mlflow, config) as ctx:
        ctx.log_prompt_version(dataset, "decoder", system_prompt)

        if mlflow.run_id is not None:
            step_offset = ctx.existing_max_step()
            logger.info("continuing run '%s' at step %d", ctx.run_id, step_offset)
        else:
            step_offset = 0

        accumulator = StreamingClassificationMetrics()

        logger.info("starting eval loop")
        with tqdm(
            desc="                             evaluating",
            unit="samples",
        ) as bar:
            n_seen, invalid = await _stream_evaluate(
                predictor=predictor,
                samples=data.test.samples(),
                concurrency=stream.concurrency,
                log_interval=stream.log_interval,
                ctx=ctx,
                accumulator=accumulator,
                bar=bar,
                step_offset=step_offset,
            )

        log_invalid(invalid, n_seen)

        items = accumulator.finalize().items()
        metrics = {f"test_{key}": value for key, value in items}
        ctx.log_metrics(metrics, step=n_seen + step_offset)

    log_summary(bar, "evaluated", n_seen, invalid)
    pretty = " ".join(f"'{key}'={value}" for key, value in sorted(metrics.items()))
    logger.info(pretty)
