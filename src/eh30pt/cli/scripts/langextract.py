"""Generate extraction predictions with langextract on a server."""

import asyncio
import json
import warnings
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

from langextract import extract
from langextract.core.types import FormatType
from returns.result import Failure, Success
from tqdm import tqdm

from eh30pt.datasets.registry import (
    assert_extraction,
    get_extract_fields,
    get_extract_shots,
    get_response_schema,
    get_sys_prompt,
    load_data,
)
from eh30pt.logging import logger
from eh30pt.openai.complete import CompletionClient
from eh30pt.openai.stream import log_invalid, log_summary, stream_predictions
from eh30pt.settings import MlflowSettings
from eh30pt.tracking.context import RunContext
from eh30pt.tracking.resume import (
    build_run_name,
)
from eh30pt.tracking.shards import log_shard
from eh30pt.types import FewShot, Predictor, asafe

if TYPE_CHECKING:
    from langextract.core.types import JsonSchema
    from langextract.data import AnnotatedDocument, ExampleData
    from langextract.factory import ModelConfig
    from transformers import PreTrainedTokenizerFast

    from eh30pt.settings import ServerSettings, StreamSettings
    from eh30pt.types import (
        DatasetName,
        ExtractShot,
        FewShotRunConfig,
        GenerateRecord,
        Method,
    )


def build_examples(shots: list[ExtractShot]) -> list[ExampleData]:
    """Build langextract examples pairing texts with field extractions."""
    from langextract.data import ExampleData, Extraction

    return [
        ExampleData(
            text=shot.user,
            extractions=[
                Extraction(extraction_class=field, extraction_text=span)
                for field, span in shot.extractions.items()
            ],
        )
        for shot in shots
    ]


def assemble_prediction(
    result: AnnotatedDocument,
    fields: tuple[str, ...],
) -> str | None:
    """Assemble the field object JSON, returning None without extractions."""
    if not result.extractions:
        return None

    grouped = {
        extraction.extraction_class: extraction.extraction_text
        for extraction in result.extractions
    }
    prediction = {field: grouped.get(field, "") for field in fields}

    return json.dumps(prediction, ensure_ascii=False, sort_keys=True)


def build_model_config(model: str, base_url: str, api_key: str) -> ModelConfig:
    """Build the OpenAI-provider config pointing langextract at the server."""
    from langextract.factory import ModelConfig

    return ModelConfig(
        model_id=model,
        provider="openai",
        provider_kwargs={"api_key": api_key, "base_url": base_url},
    )


async def _predict_one(
    user: str,
    *,
    system_prompt: str,
    examples: list[ExampleData],
    model_config: ModelConfig,
    fields: tuple[str, ...],
    output_schema: JsonSchema,
    seed: int | None,
) -> str | None:
    """Extract one field object, returning None when the call fails."""
    match await _extract_one(
        user,
        system_prompt,
        examples,
        model_config,
        output_schema=output_schema,
        seed=seed,
    ):
        case Success(result):
            return assemble_prediction(result, fields)
        case Failure(exc):
            logger.error("extraction failed: %s", exc)
            return None


@asafe
async def _extract_one(
    text: str,
    prompt_description: str,
    examples: list[ExampleData],
    config: ModelConfig,
    *,
    output_schema: JsonSchema,
    seed: int | None,
) -> AnnotatedDocument:
    """Run langextract on a single text without blocking the event loop."""
    language_model_params = {"seed": seed} if seed is not None else None

    # Schema constraints come from the dataset schema, examples guide phrasing.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="With 'config', schema constraints.*",
            category=UserWarning,
        )

        result = await asyncio.to_thread(
            extract,
            text,
            prompt_description=prompt_description,
            examples=examples,
            config=config,
            format_type=FormatType.JSON,
            output_schema=output_schema,
            language_model_params=language_model_params,
            show_progress=False,
        )

    return cast("AnnotatedDocument", result)


@dataclass(frozen=True)
class ExtractPredictor(Predictor[FewShot, str | None]):
    """Bind langextract inputs for streaming field-object extraction."""

    system_prompt: str
    examples: list[ExampleData]
    model_config: ModelConfig
    fields: tuple[str, ...]
    output_schema: JsonSchema
    seed: int | None

    async def predict(self, sample: FewShot) -> str | None:
        """Extract one streamed sample, returning None on empty runs."""
        prediction = await _predict_one(
            sample.user,
            system_prompt=self.system_prompt,
            examples=self.examples,
            model_config=self.model_config,
            fields=self.fields,
            output_schema=self.output_schema,
            seed=self.seed,
        )

        if prediction is None:
            logger.debug("invalid response for sample %s", sample.id)

        return prediction


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
    """Generate predictions with langextract and log them as sharded artifacts."""
    assert_extraction(dataset, "langextract needs extraction fields")

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

    method: Method = "langextract"
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

    examples = build_examples(get_extract_shots(dataset, n_shot))
    fields = get_extract_fields(dataset)
    output_schema = cast(
        "JsonSchema",
        get_response_schema(dataset)["json_schema"]["schema"],
    )

    model_config = build_model_config(model, server.base_url, server.api_key)

    predictor = ExtractPredictor(
        system_prompt=system_prompt,
        examples=examples,
        model_config=model_config,
        fields=fields,
        output_schema=output_schema,
        seed=seed,
    )

    with TemporaryDirectory() as tmp, RunContext(mlflow, config) as ctx:
        ctx.log_prompt_version(dataset, "decoder", system_prompt)

        if mlflow.run_id is not None:
            start_shard = ctx.next_shard_index()
            logger.info("continuing run '%s' at shard %d", ctx.run_id, start_shard)
        else:
            start_shard = 0

        batch: list[GenerateRecord] = []

        logger.info("starting langextract loop")
        with tqdm(desc="extracting", unit="samples") as bar:
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

        log_invalid(invalid, n_seen)

        if batch:
            n_shards = log_shard(
                ctx,
                tmp,
                MlflowSettings.predictions_path,
                n_shards,
                batch,
            )

            batch.clear()

    log_summary(bar, "extracted", n_seen, invalid, n_shards=n_shards)
    logger.info("logged %d shards to '%s'", n_shards, ctx.artifact_uri)
