"""Run few-shot test evaluation against a llama server."""

import json
from typing import TYPE_CHECKING, NamedTuple, cast

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from eh30pt.cli.scripts.common import log_prompt_version
from eh30pt.cli.scripts.completion import complete, server_model
from eh30pt.datasets.registry import get_dataset, get_labels
from eh30pt.logging import logger
from eh30pt.tracking import build_run_name, open_run

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam
    from transformers import PreTrainedTokenizerFast

    from eh30pt.types import DatasetName, EvalExample, FewShotRunConfig


class EvalData(NamedTuple):
    """Few-shot examples and test samples for API-side evaluation."""

    examples: list[EvalExample]
    samples: list[EvalExample]
    labels: tuple[str, ...]


def load_eval_data(dataset: DatasetName, n_shot: int) -> EvalData:
    """Load few-shot examples and test samples for dataset."""
    tokenizer = cast("PreTrainedTokenizerFast", None)
    data = get_dataset(dataset, tokenizer, "decoder")
    data.load_raw()

    samples = data.eval_samples()
    return EvalData(
        examples=data.few_shot_examples(n_shot),
        samples=samples,
        labels=get_labels(dataset),
    )


def build_converstaion(examples: list[EvalExample]) -> list[ChatCompletionMessageParam]:
    """Format few-shot examples as alternating user and assistant turns."""
    conversation: list[ChatCompletionMessageParam] = []
    for example in examples:
        answer = json.dumps({"label": example.label})
        conversation.append({"role": "user", "content": example.user})
        conversation.append({"role": "assistant", "content": answer})

    return conversation


def compute_metrics(references: list[str], predictions: list[str]) -> dict[str, float]:
    """Compute accuracy and macro-averaged precision, recall, and f1."""
    accuracy = accuracy_score(references, predictions)
    precision = precision_score(
        references,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        references,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(references, predictions, average="macro", zero_division=0)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


async def main(
    *,
    system_prompt: str,
    base_url: str,
    dataset: DatasetName,
    n_shot: int,
    one_conv: bool = False,
    mlflow_run_name: str | None,
    mlflow_experiment: str,
    mlflow_tracking_uri: str,
    seed: int | None,
) -> None:
    """Run few-shot test evaluation on the llama server."""
    data = load_eval_data(dataset, n_shot)
    if not data.samples:
        msg = f"no test samples for dataset '{dataset}'"
        raise ValueError(msg)

    model = await server_model(base_url)
    method = f"{n_shot}-shot"
    if mlflow_run_name is None:
        mlflow_run_name = build_run_name(dataset, model, method, train_samples=0)

    logger.info("tracking '%s' of experiment '%s'", mlflow_run_name, mlflow_experiment)
    run_config: FewShotRunConfig = {
        "method": method,
        "dataset": dataset,
        "architecture": "decoder",
        "base_model": model,
        "n_shot": n_shot,
        "batch_size": 1,
        "seed": seed,
        "one_conv": one_conv,
        "system_prompt": system_prompt,
        "train_samples": 0,
        "validation_samples": 0,
        "test_samples": len(data.samples),
        "total_parameters": 0,
        "trainable_parameters": 0,
    }

    conversation: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        *build_converstaion(data.examples),
    ]

    references: list[str] = []
    predictions: list[str] = []
    invalid = 0

    total = len(data.samples)

    logger.info("evaluating %d samples with %d examples", total, n_shot)
    samples = tqdm(data.samples, desc="evaluating", unit="sample")
    for index, sample in enumerate(samples, start=1):
        messages: list[ChatCompletionMessageParam]
        if one_conv:
            conversation.append({"role": "user", "content": sample.user})
            messages = conversation
        else:
            messages = [*conversation, {"role": "user", "content": sample.user}]

        try:
            prediction = await complete(
                base_url,
                model,
                messages,
                labels=data.labels,
                seed=seed,
            )
        except ValueError:
            invalid += 1
            logger.debug("invalid response for sample %d, counting as other", index)
            prediction = "other"

        if one_conv:
            answer = json.dumps({"label": prediction})
            conversation.append({"role": "assistant", "content": answer})

        references.append(sample.label)
        predictions.append(prediction)

    if invalid:
        logger.warning("%d/%d invalid responses counted as other", invalid, total)

    metrics = {
        f"test_{key}": value
        for key, value in compute_metrics(references, predictions).items()
    }

    with open_run(
        mlflow_experiment,
        mlflow_run_name,
        mlflow_tracking_uri,
        resume=False,
        config=run_config,
    ) as ctx:
        log_prompt_version(ctx, dataset, "decoder", system_prompt)
        ctx.log_metrics(metrics)

    logger.info(" ".join(f"'{key}'={value}" for key, value in sorted(metrics.items())))
