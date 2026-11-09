"""Generate label-constrained predictions on a server."""

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

from tqdm import tqdm

from eh30pt.datasets.registry import (
    assert_classification,
    get_few_shots,
    get_labels,
    get_sys_prompt,
    load_data,
)
from eh30pt.logging import logger
from eh30pt.openai.complete import OpenaiClient
from eh30pt.openai.stream import (
    OpenaiPredictor,
    log_summary,
    maybe_log_invalid,
    stream_predictions,
)
from eh30pt.settings import MlflowSettings
from eh30pt.systemone.client import SystemOneClient
from eh30pt.systemone.stream import SystemOnePredictor
from eh30pt.tracking.context import RunContext
from eh30pt.tracking.resume import (
    build_run_name,
)
from eh30pt.tracking.shards import log_shard

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from eh30pt.settings import ServerSettings, StreamSettings
    from eh30pt.types import (
        DatasetName,
        FewShot,
        FewShotRunConfig,
        GenerateRecord,
        Method,
        Predictor,
    )


async def main(
    *,
    system_prompt: str | None,
    server: ServerSettings,
    dataset: DatasetName,
    n_shot: int,
    stream: StreamSettings,
    mlflow: MlflowSettings,
    seed: int | None,
    systemone: bool = False,
) -> None:
    """Generate predictions on the server and log them as sharded artifacts."""
    if systemone:
        assert_classification(dataset, "System One requires classification labels")

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

    predictor: Predictor[FewShot, str | None]
    if systemone:
        client = SystemOneClient(server)
        model = (await client.model()).unwrap()
        labels = get_labels(dataset)
        examples = get_few_shots(dataset, n_shot)
        predictor = SystemOnePredictor(client, system_prompt, labels, examples)
    else:
        client = OpenaiClient(server, seed)
        model = (await client.model()).unwrap()
        examples = data.test.examples
        conv = client.build_conversation(system=system_prompt, examples=examples)
        predictor = OpenaiPredictor(client=client, conversation=conv)

    method: Method = "systemone" if systemone else "few-shot"
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

    with TemporaryDirectory() as tmp, RunContext(mlflow, config) as ctx:
        ctx.log_prompt_version(dataset, "decoder", system_prompt)

        if mlflow.run_id is not None:
            start_shard = ctx.next_shard_index()
            logger.info("continuing run '%s' at shard %d", ctx.run_id, start_shard)
        else:
            start_shard = 0

        batch: list[GenerateRecord] = []

        logger.info("starting generate loop")
        with tqdm(desc="Generating", total=data.test.total()) as bar:
            n_seen, invalid, n_shards = await stream_predictions(
                data.test.samples(),
                predictor,
                concurrency=stream.concurrency,
                shard_size=stream.shard_size,
                directory=tmp,
                ctx=ctx,
                batch=batch,
                bar=bar,
                start_shard=start_shard,
            )

        maybe_log_invalid(invalid, n_seen)

        if batch:
            n_shards = log_shard(
                ctx,
                tmp,
                MlflowSettings.predictions_path,
                n_shards,
                batch,
            )

            batch.clear()

    log_summary(bar, "generated", n_seen, invalid, n_shards=n_shards)
    logger.info("logged %d shards to '%s'", n_shards, ctx.artifact_uri)
