"""Streaming evaluation, export, and predictor integration tests."""

from __future__ import annotations

import asyncio
import io
import json
from typing import TYPE_CHECKING, cast

import mlflow
import pytest
from langextract.data import AnnotatedDocument, Extraction
from returns.result import Failure, Success
from tqdm.rich import tqdm

import eh30pt.cli.scripts.langextract as langextract_mod
from eh30pt.cli.scripts.create_task import run_labeling
from eh30pt.datasets.discont.extract import DiscontExtract
from eh30pt.datasets.registry import get_extract_fields, get_sys_prompt
from eh30pt.modeling.metrics import StreamingClassificationMetrics
from eh30pt.openai.complete import CompletionClient
from eh30pt.openai.stream import CompletionPredictor
from eh30pt.settings import MlflowSettings
from eh30pt.tracking.context import RunContext
from eh30pt.tracking.shards import log_shard
from eh30pt.types import FewShot, GenerateRecord, Predictor

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from langextract.core.types import JsonSchema
    from mlflow.tracking import MlflowClient
    from openai.types.chat import ChatCompletionMessageParam
    from returns.result import Result

    from eh30pt.labeling.types import ChoiceValue, SpanValue
    from eh30pt.types import ExtractShot, RunConfig


@pytest.fixture
def mlflow_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MlflowClient]:
    """Point fluent tracking at an isolated sqlite store."""
    from mlflow.tracking import MlflowClient

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)

    yield client

    if mlflow.active_run() is not None:
        mlflow.end_run()


@pytest.fixture
def run_config() -> RunConfig:
    """Return a minimal few-shot run config."""
    from eh30pt.datasets.registry import get_sys_prompt
    from test.conftest import models

    return {
        "method": "few-shot",
        "dataset": "oblique",
        "architecture": "decoder",
        "base_model": models["gpt2"],
        "n_shot": 0,
        "batch_size": 1,
        "system_prompt": get_sys_prompt("oblique"),
    }


class _FailingClient(CompletionClient):
    """Completion client stub whose complete always fails."""

    def __init__(self) -> None:
        """Skip endpoint setup; complete never reaches the network."""

    async def complete(
        self, conversation: list[ChatCompletionMessageParam]
    ) -> Result[str, Exception]:
        _ = conversation
        return Failure(RuntimeError("no server"))


class _EchoClient(CompletionClient):
    """Completion client stub echoing the last user message."""

    def __init__(self) -> None:
        """Skip endpoint setup; complete never reaches the network."""

    async def complete(
        self, conversation: list[ChatCompletionMessageParam]
    ) -> Result[str, Exception]:
        last = cast("dict[str, object]", conversation[-1])
        content = last.get("content")
        return Success(content if isinstance(content, str) else "")


def test_completion_predictor_returns_none_on_failure() -> None:
    from eh30pt.datasets.registry import get_few_shots

    sample = get_few_shots("oblique", 1)[0]
    predictor = CompletionPredictor(client=_FailingClient(), conversation=[])
    result = asyncio.run(predictor.predict(sample))

    assert result is None


def test_completion_predictor_returns_content_on_success() -> None:
    from eh30pt.datasets.registry import get_few_shots

    sample = get_few_shots("oblique", 1)[0]
    predictor = CompletionPredictor(client=_EchoClient(), conversation=[])
    result = asyncio.run(predictor.predict(sample))

    assert result == sample.user


def test_completion_script_logs_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import eh30pt.cli.scripts.completion as completion_mod
    from eh30pt.datasets.registry import get_few_shots, get_sys_prompt
    from eh30pt.settings import ServerSettings

    sample = get_few_shots("oblique", 1)[0]
    system = get_sys_prompt("oblique")

    async def _ok(self: object, conversation: object) -> Success[str]:
        _ = self, conversation
        return Success(sample.label)

    monkeypatch.setattr(completion_mod.CompletionClient, "complete", _ok)
    with caplog.at_level("INFO", logger="eh30pt"):
        asyncio.run(
            completion_mod.main(
                user_prompt=sample.user,
                system_prompt=system,
                server=ServerSettings(),
            )
        )

    assert f"completion: {sample.label}" in caplog.text
    caplog.clear()

    async def _boom(self: object, conversation: object) -> Result[str, Exception]:
        _ = self, conversation
        return Failure(RuntimeError("boom"))

    monkeypatch.setattr(completion_mod.CompletionClient, "complete", _boom)
    with caplog.at_level("ERROR", logger="eh30pt"):
        asyncio.run(
            completion_mod.main(
                user_prompt=sample.user,
                system_prompt=system,
                server=ServerSettings(),
            )
        )

    assert "completion failed" in caplog.text


def test_predict_one_assembles_and_handles_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shot = DiscontExtract.extract_shots[0]
    fields = get_extract_fields("discont-extract/trigger")
    system = get_sys_prompt("discont-extract/trigger")
    doc = AnnotatedDocument(
        text=shot.user,
        extractions=[
            Extraction(
                extraction_class="drug", extraction_text=shot.extractions["drug"]
            )
        ],
    )
    model_config = langextract_mod.build_model_config("m", "http://x", "k")
    output_schema = cast("JsonSchema", DiscontExtract.response_format())

    async def _ok(
        *args: object, **kwargs: object
    ) -> Result[AnnotatedDocument, Exception]:
        _ = args, kwargs
        return Success(doc)

    monkeypatch.setattr(langextract_mod, "_extract_one", _ok)
    result = asyncio.run(
        langextract_mod._predict_one(
            shot.user,
            system_prompt=system,
            examples=[],
            model_config=model_config,
            fields=fields,
            output_schema=output_schema,
            seed=None,
        )
    )

    assert result is not None
    assert json.loads(result) == {
        "phrase": "",
        "reason": "",
        "drug": shot.extractions["drug"],
    }

    async def _boom(
        *args: object, **kwargs: object
    ) -> Result[AnnotatedDocument, Exception]:
        _ = args, kwargs
        return Failure(RuntimeError("boom"))

    monkeypatch.setattr(langextract_mod, "_extract_one", _boom)
    result = asyncio.run(
        langextract_mod._predict_one(
            shot.user,
            system_prompt=system,
            examples=[],
            model_config=model_config,
            fields=(fields[0],),
            output_schema=output_schema,
            seed=None,
        )
    )

    assert result is None


def test_extract_predictor_predicts_and_marks_invalid(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    shot = DiscontExtract.extract_shots[0]
    fields = get_extract_fields("discont-extract/trigger")
    system = get_sys_prompt("discont-extract/trigger")
    sample = FewShot(user=shot.user, label="")
    doc = AnnotatedDocument(
        text=shot.user,
        extractions=[
            Extraction(
                extraction_class="drug", extraction_text=shot.extractions["drug"]
            )
        ],
    )
    model_config = langextract_mod.build_model_config("m", "http://x", "k")
    output_schema = cast("JsonSchema", DiscontExtract.response_format())

    async def _ok(
        *args: object, **kwargs: object
    ) -> Result[AnnotatedDocument, Exception]:
        _ = args, kwargs
        return Success(doc)

    monkeypatch.setattr(langextract_mod, "_extract_one", _ok)
    predictor = langextract_mod.ExtractPredictor(
        system_prompt=system,
        examples=[],
        model_config=model_config,
        fields=(fields[0],),
        output_schema=output_schema,
        seed=None,
    )
    result = asyncio.run(predictor.predict(sample))

    assert result is not None

    async def _boom(
        *args: object, **kwargs: object
    ) -> Result[AnnotatedDocument, Exception]:
        _ = args, kwargs
        return Failure(RuntimeError("boom"))

    monkeypatch.setattr(langextract_mod, "_extract_one", _boom)
    with caplog.at_level("DEBUG", logger="eh30pt"):
        result = asyncio.run(predictor.predict(sample))

    assert result is None


class _MapPredictor(Predictor[FewShot, str | None]):
    """Predictor returning the gold label, or None for empty users."""

    async def predict(self, sample: FewShot) -> str | None:
        if not sample.user:
            return None

        return sample.label


def test_fold_eval_window_counts_and_logs(
    mlflow_client: MlflowClient, run_config: RunConfig
) -> None:
    from eh30pt.cli.scripts.evaluate import _fold_eval_window
    from eh30pt.datasets.registry import get_few_shots

    _ = mlflow_client
    mlflow.set_experiment("fold-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="fold-exp"
    )
    shots = get_few_shots("oblique", 3)
    with RunContext(settings, config=run_config) as ctx:
        accumulator = StreamingClassificationMetrics()
        window = [
            shots[0],
            FewShot(user="", label=shots[1].label),
            shots[1],
        ]

        n_seen, invalid = _fold_eval_window(
            ctx,
            accumulator,
            window,
            [shots[0].label, None, shots[2].label],
            log_interval=2,
            n_seen=0,
            invalid=0,
        )

        assert (n_seen, invalid) == (3, 1)
        assert accumulator.finalize()["accuracy"] == pytest.approx(0.5)

        logged, _ = _fold_eval_window(
            ctx,
            StreamingClassificationMetrics(),
            [shots[0], shots[1]],
            [shots[0].label, shots[1].label],
            log_interval=2,
            n_seen=0,
            invalid=0,
        )

        _expected_seen = 2
        assert logged == _expected_seen


def test_stream_evaluate_consumes_windows(
    mlflow_client: MlflowClient, run_config: RunConfig
) -> None:
    from eh30pt.cli.scripts.evaluate import _stream_evaluate
    from eh30pt.datasets.registry import get_few_shots

    _ = mlflow_client
    mlflow.set_experiment("stream-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="stream-exp"
    )
    shots = get_few_shots("oblique", 2)
    samples = [
        shots[0],
        FewShot(user="", label=shots[1].label),
        shots[1],
    ]
    with RunContext(settings, config=run_config) as ctx:
        bar = tqdm(
            desc="                             evaluating",
            unit="samples",
            file=io.StringIO(),
        )
        try:
            n_seen, invalid = asyncio.run(
                _stream_evaluate(
                    predictor=_MapPredictor(),
                    samples=iter(samples),
                    concurrency=2,
                    log_interval=1000,
                    ctx=ctx,
                    accumulator=StreamingClassificationMetrics(),
                    bar=bar,
                )
            )
        finally:
            bar.close()

        assert (n_seen, invalid) == (3, 1)


def _span_batch() -> tuple[ExtractShot, ExtractShot, list[GenerateRecord]]:
    valid = DiscontExtract.extract_shots[0]
    empty_user = DiscontExtract.extract_shots[4].user
    other = DiscontExtract.extract_shots[1]
    third = DiscontExtract.extract_shots[3]
    phrase = valid.extractions["phrase"]
    batch = [
        GenerateRecord(
            id=0,
            user=valid.user,
            prediction=json.dumps(
                valid.extractions, ensure_ascii=False, sort_keys=True
            ),
        ),
        GenerateRecord(
            id=1,
            user=empty_user,
            prediction=json.dumps({"phrase": "", "reason": "", "drug": ""}),
        ),
        GenerateRecord(
            id=2,
            user=other.user,
            prediction=json.dumps({"phrase": phrase, "reason": "x", "drug": "y"}),
        ),
        GenerateRecord(
            id=3,
            user=third.user,
            prediction=json.dumps(
                third.extractions, ensure_ascii=False, sort_keys=True
            ),
        ),
        GenerateRecord(
            id=4,
            user=other.user,
            prediction=json.dumps(
                {"phrase": valid.extractions["phrase"], "reason": "r", "drug": "d"}
            ),
        ),
    ]
    return valid, other, batch


def test_run_labeling_builds_span_tasks(
    mlflow_client: MlflowClient, run_config: RunConfig, tmp_path: Path
) -> None:
    from eh30pt.cli.scripts.create_task import run_labeling
    from eh30pt.labeling.tasks.extract import _locate_span

    _ = mlflow_client
    mlflow.set_experiment("export-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="export-exp"
    )
    valid, other, batch = _span_batch()
    user = valid.user
    phrase = valid.extractions["phrase"]
    reason = valid.extractions["reason"]
    drug = valid.extractions["drug"]
    with RunContext(settings, config=run_config) as ctx:
        run_id = ctx.run_id
        log_shard(ctx, tmp_path, MlflowSettings.predictions_path, 0, batch)

    labeling = run_labeling(
        mlflow_settings=MlflowSettings(
            tracking_uri=mlflow_client.tracking_uri,
            run_id=run_id,
        ),
        dataset="discont-extract/trigger",
    )

    tasks = labeling.tasks
    config = labeling.config

    assert len(tasks) == 1
    assert tasks[0]["data"] == {
        "text": user,
        "phrase": phrase,
        "reason": reason,
        "drug": drug,
    }

    start, end = user.index(phrase), user.index(phrase) + len(phrase)
    reason_start = user.index(reason)
    drug_start = user.index(drug)
    assert _locate_span(user, phrase) == (start, end)
    assert _locate_span(user, "") is None
    assert _locate_span(user, other.user) is None
    assert tasks[0]["predictions"] == [
        {
            "result": [
                {
                    "from_name": "span",
                    "to_name": "text",
                    "type": "labels",
                    "readonly": True,
                    "hidden": False,
                    "value": {
                        "start": start,
                        "end": end,
                        "text": phrase,
                        "labels": ["phrase"],
                    },
                },
                {
                    "from_name": "span",
                    "to_name": "text",
                    "type": "labels",
                    "readonly": True,
                    "hidden": False,
                    "value": {
                        "start": reason_start,
                        "end": reason_start + len(reason),
                        "text": reason,
                        "labels": ["reason"],
                    },
                },
                {
                    "from_name": "span",
                    "to_name": "text",
                    "type": "labels",
                    "readonly": True,
                    "hidden": False,
                    "value": {
                        "start": drug_start,
                        "end": drug_start + len(drug),
                        "text": drug,
                        "labels": ["drug"],
                    },
                },
            ],
            "model_version": run_id,
        }
    ]

    assert '<Text name="text" value="$text" />' in config
    assert '<View style="display: none">' in config
    assert '<Labels name="span" toName="text">' in config
    assert '<Label value="phrase" />' not in config
    assert '<Label value="reason" />' not in config
    assert '<Label value="drug" />' not in config
    assert 'value="phrase" background="#ffe08a"' in config
    assert 'value="reason" background="#a9c6ff"' in config
    assert 'value="drug" background="#a8e6a3"' in config
    assert config.index("<Choices ") < config.index('<View style="display: none">')

    assert '<Header value="Drug discontinuation" />' in config
    assert '<Header value="Not discontinuation" />' in config
    assert '<Header value="Meta" />' in config

    for name in ("discont", "other", "meta"):
        choices = f'<Choices name="{name}" toName="text" choice="multiple"'
        assert f'{choices} showInLine="false">' in config

    discontinuation = (
        "Treatment success",
        "Treatment inefficacy",
        "Contraindication/unsafe co-usage",
        "Adverse reactions",
        "Other",
    )

    other = (
        "Adverse reaction without discontinuation",
        "Treatment inefficacy without discontinuation",
        "Dosage change",
        "Other non-discontinuation",
    )

    meta = (
        "Incomplete phrase",
        "Incorrect phrase",
    )

    for label in (*discontinuation, *other, *meta):
        assert f'value="{label}"' in config
        assert f'value="{label}" hint=' in config

    for label in ("Complex fact", "Non-medical reasons", "Indeterminate"):
        assert f'value="{label}"' not in config


def test_run_labeling_builds_choice_tasks_for_classification_dataset(
    mlflow_client: MlflowClient, run_config: RunConfig, tmp_path: Path
) -> None:
    from eh30pt.cli.scripts.create_task import run_labeling
    from eh30pt.datasets.registry import get_few_shots, get_labels

    _ = mlflow_client
    mlflow.set_experiment("export-oblique-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="export-oblique-exp"
    )
    shots = get_few_shots("oblique", 2)
    invalid_label = get_labels("multinerd")[0]
    with RunContext(settings, config=run_config) as ctx:
        run_id = ctx.run_id
        batch = [
            GenerateRecord(id=0, user=shots[0].user, prediction=shots[0].label),
            GenerateRecord(id=1, user=shots[1].user, prediction=shots[1].label),
            GenerateRecord(id=2, user=shots[0].user, prediction=invalid_label),
        ]
        log_shard(ctx, tmp_path, MlflowSettings.predictions_path, 0, batch)

    labeling = run_labeling(
        mlflow_settings=MlflowSettings(
            tracking_uri=mlflow_client.tracking_uri,
            run_id=run_id,
        ),
        dataset="oblique",
    )

    tasks = labeling.tasks
    config = labeling.config

    _exported = 2
    assert len(tasks) == _exported
    assert tasks[0] == {
        "data": {"text": shots[0].user},
        "predictions": [
            {
                "result": [
                    {
                        "from_name": "label",
                        "to_name": "text",
                        "type": "choices",
                        "readonly": True,
                        "hidden": False,
                        "value": {"choices": [shots[0].label]},
                    }
                ],
                "model_version": run_id,
            }
        ],
    }
    chosen = tasks[1]["predictions"][0]["result"][0]["value"]
    assert cast("ChoiceValue", chosen)["choices"] == [shots[1].label]

    assert '<Text name="text" value="$text" />' in config
    assert (
        '<Choices name="label" toName="text" choice="single" showInLine="true">'
        in config
    )
    for label in get_labels("oblique"):
        assert f'<Choice value="{label}" />' in config

    assert 'value="phrase" background="#ffe08a"' not in config
    assert '<View style="display: none">' not in config


def test_run_labeling_reads_struct_predictions(
    mlflow_client: MlflowClient, run_config: RunConfig, tmp_path: Path
) -> None:
    _ = mlflow_client
    mlflow.set_experiment("export-struct-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="export-struct-exp"
    )
    shot = DiscontExtract.extract_shots[1]
    user = shot.user
    phrase = shot.extractions["phrase"]
    with RunContext(settings, config=run_config) as ctx:
        run_id = ctx.run_id
        log_shard(
            ctx,
            tmp_path,
            MlflowSettings.predictions_path,
            0,
            [
                {
                    "id": 0,
                    "user": user,
                    "prediction": dict(shot.extractions),
                },
                {
                    "id": 1,
                    "user": DiscontExtract.extract_shots[4].user,
                    "prediction": {"phrase": "", "reason": "", "drug": ""},
                },
            ],
        )

    labeling = run_labeling(
        mlflow_settings=MlflowSettings(
            tracking_uri=mlflow_client.tracking_uri,
            run_id=run_id,
        ),
        dataset="discont-extract/trigger",
    )

    tasks = labeling.tasks

    assert len(tasks) == 1
    assert tasks[0]["data"]["text"] == user

    start = user.index(phrase)
    span = cast("SpanValue", tasks[0]["predictions"][0]["result"][0]["value"])
    assert (span["start"], span["end"]) == (start, start + len(phrase))
    assert tasks[0]["predictions"][0]["model_version"] == run_id


def test_run_labeling_rejects_non_extraction_predictions(
    mlflow_client: MlflowClient, run_config: RunConfig, tmp_path: Path
) -> None:
    from eh30pt.cli.scripts.create_task import run_labeling
    from eh30pt.datasets.registry import get_few_shots, get_labels

    _ = mlflow_client
    mlflow.set_experiment("export-plain-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="export-plain-exp"
    )
    sample = get_few_shots("oblique", 1)[0]
    with RunContext(settings, config=run_config) as ctx:
        run_id = ctx.run_id
        batch = [
            GenerateRecord(
                id=0,
                user=sample.user,
                prediction=get_labels("discont-classifier/extract-labels")[0],
            )
        ]
        log_shard(ctx, tmp_path, MlflowSettings.predictions_path, 0, batch)

    with pytest.raises(ValueError, match="not extraction JSON"):
        run_labeling(
            mlflow_settings=MlflowSettings(
                tracking_uri=mlflow_client.tracking_uri,
                run_id=run_id,
            ),
            dataset="discont-extract/trigger",
        )


def test_run_labeling_rejects_non_object_predictions(
    mlflow_client: MlflowClient, run_config: RunConfig, tmp_path: Path
) -> None:
    from eh30pt.cli.scripts.create_task import run_labeling
    from eh30pt.datasets.registry import get_few_shots

    _ = mlflow_client
    mlflow.set_experiment("export-number-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="export-number-exp"
    )
    sample = get_few_shots("oblique", 1)[0]
    with RunContext(settings, config=run_config) as ctx:
        run_id = ctx.run_id
        batch = [GenerateRecord(id=0, user=sample.user, prediction="123")]
        log_shard(ctx, tmp_path, MlflowSettings.predictions_path, 0, batch)

    with pytest.raises(ValueError, match="not extraction JSON"):
        run_labeling(
            mlflow_settings=MlflowSettings(
                tracking_uri=mlflow_client.tracking_uri,
                run_id=run_id,
            ),
            dataset="discont-extract/trigger",
        )


def test_run_labeling_without_shards_is_empty(
    mlflow_client: MlflowClient, run_config: RunConfig
) -> None:
    from eh30pt.cli.scripts.create_task import run_labeling

    _ = mlflow_client
    mlflow.set_experiment("export-empty-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="export-empty-exp"
    )
    with RunContext(settings, config=run_config) as ctx:
        run_id = ctx.run_id

    labeling = run_labeling(
        mlflow_settings=MlflowSettings(
            tracking_uri=mlflow_client.tracking_uri,
            run_id=run_id,
        ),
        dataset="discont-extract/trigger",
    )

    assert labeling.tasks == []


def test_run_labeling_without_run_id_raises() -> None:
    from eh30pt.cli.scripts.create_task import run_labeling

    with pytest.raises(ValueError, match="no mlflow run id"):
        run_labeling(
            mlflow_settings=MlflowSettings(tracking_uri="sqlite:///mlflow.db"),
            dataset="discont-extract/trigger",
        )


def test_run_labeling_infers_dataset_from_run(
    mlflow_client: MlflowClient, run_config: RunConfig, tmp_path: Path
) -> None:
    from eh30pt.cli.scripts.create_task import run_labeling
    from eh30pt.datasets.registry import get_few_shots

    _ = mlflow_client
    mlflow.set_experiment("export-infer-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="export-infer-exp"
    )
    shots = get_few_shots("oblique", 2)
    with RunContext(settings, config=run_config) as ctx:
        run_id = ctx.run_id
        batch = [
            GenerateRecord(id=0, user=shots[0].user, prediction=shots[0].label),
            GenerateRecord(id=1, user=shots[1].user, prediction=shots[1].label),
        ]
        log_shard(ctx, tmp_path, MlflowSettings.predictions_path, 0, batch)

    labeling = run_labeling(
        mlflow_settings=MlflowSettings(
            tracking_uri=mlflow_client.tracking_uri,
            run_id=run_id,
        ),
    )

    assert [task["data"] for task in labeling.tasks] == [
        {"text": shots[0].user},
        {"text": shots[1].user},
    ]


def test_run_labeling_without_dataset_in_run_raises(
    mlflow_client: MlflowClient,
) -> None:
    from eh30pt.cli.scripts.create_task import run_labeling

    _ = mlflow_client
    mlflow.set_experiment("export-no-dataset-exp")
    with mlflow.start_run() as run:
        run_id = run.info.run_id

    with pytest.raises(ValueError, match="records no dataset"):
        run_labeling(
            mlflow_settings=MlflowSettings(
                tracking_uri=mlflow_client.tracking_uri,
                run_id=run_id,
            ),
        )


def test_model_raises_when_server_serves_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from eh30pt.openai.complete import CompletionClient
    from eh30pt.settings import ServerSettings

    async def _empty() -> object:
        return SimpleNamespace(data=[])

    client = CompletionClient(ServerSettings())
    monkeypatch.setattr(
        client,
        "_client",
        SimpleNamespace(base_url="test", models=SimpleNamespace(list=_empty)),
    )
    result = asyncio.run(client.model())

    assert isinstance(result, Failure)
    assert "serves no models" in str(result.failure())


def test_model_caches_first_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from openai.types import Model

    from eh30pt.openai.complete import CompletionClient
    from eh30pt.settings import ServerSettings

    calls = 0

    async def _page() -> object:
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            data=[Model(id="org/llama", created=0, object="model", owned_by="org")]
        )

    client = CompletionClient(ServerSettings())
    monkeypatch.setattr(
        client,
        "_client",
        SimpleNamespace(base_url="test", models=SimpleNamespace(list=_page)),
    )

    assert asyncio.run(client.model()).unwrap() == "llama"
    assert asyncio.run(client.model()).unwrap() == "llama"
    assert calls == 1


def test_langextract_main_rejects_classification_dataset() -> None:
    import eh30pt.cli.scripts.langextract as langextract_mod
    from eh30pt.settings import MlflowSettings, ServerSettings, StreamSettings

    with pytest.raises(ValueError, match="classification dataset"):
        asyncio.run(
            langextract_mod.main(
                system_prompt=None,
                server=ServerSettings(),
                dataset="oblique",
                n_shot=1,
                stream=StreamSettings(),
                mlflow=MlflowSettings(),
                seed=None,
            )
        )
