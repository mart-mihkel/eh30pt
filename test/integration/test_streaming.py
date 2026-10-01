"""Streaming evaluation, export, and predictor integration tests."""

from __future__ import annotations

import asyncio
import io
import json
from typing import TYPE_CHECKING, cast

import mlflow
import pytest
from returns.result import Failure, Success
from tqdm import tqdm

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

    from eh30pt.types import RunConfig


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
    return {
        "method": "few-shot",
        "dataset": "oblique",
        "architecture": "decoder",
        "base_model": "some-model",
        "n_shot": 0,
        "batch_size": 1,
        "system_prompt": "prompt",
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
    predictor = CompletionPredictor(client=_FailingClient(), conversation=[])
    result = asyncio.run(predictor.predict(FewShot(user="hi", label="hi")))

    assert result is None


def test_completion_predictor_returns_content_on_success() -> None:
    predictor = CompletionPredictor(client=_EchoClient(), conversation=[])
    result = asyncio.run(predictor.predict(FewShot(user="hi", label="hi")))

    assert result == "hi"


def test_completion_script_logs_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import eh30pt.cli.scripts.completion as completion_mod
    from eh30pt.settings import ServerSettings

    async def _ok(self: object, conversation: object) -> Success[str]:
        _ = self, conversation
        return Success("Tere")

    monkeypatch.setattr(completion_mod.CompletionClient, "complete", _ok)
    with caplog.at_level("INFO", logger="eh30pt"):
        asyncio.run(
            completion_mod.main(
                user_prompt="hi",
                system_prompt="be brief",
                server=ServerSettings(),
            )
        )

    assert "completion: Tere" in caplog.text
    caplog.clear()

    async def _boom(self: object, conversation: object) -> Result[str, Exception]:
        _ = self, conversation
        return Failure(RuntimeError("boom"))

    monkeypatch.setattr(completion_mod.CompletionClient, "complete", _boom)
    with caplog.at_level("ERROR", logger="eh30pt"):
        asyncio.run(
            completion_mod.main(
                user_prompt="hi",
                system_prompt="be brief",
                server=ServerSettings(),
            )
        )

    assert "completion failed" in caplog.text


def test_predict_one_assembles_and_handles_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langextract.data import AnnotatedDocument, Extraction

    import eh30pt.cli.scripts.langextract as langextract_mod

    doc = AnnotatedDocument(
        text="input",
        extractions=[Extraction(extraction_class="drug", extraction_text="Nebilet")],
    )
    model_config = langextract_mod.build_model_config("m", "http://x", "k")
    output_schema = cast("JsonSchema", {"type": "object"})

    async def _ok(
        *args: object, **kwargs: object
    ) -> Result[AnnotatedDocument, Exception]:
        _ = args, kwargs
        return Success(doc)

    monkeypatch.setattr(langextract_mod, "_extract_one", _ok)
    result = asyncio.run(
        langextract_mod._predict_one(
            "input",
            system_prompt="sys",
            examples=[],
            model_config=model_config,
            fields=("drug", "reason"),
            output_schema=output_schema,
            seed=None,
        )
    )

    assert result is not None
    assert json.loads(result) == {"drug": "Nebilet", "reason": ""}

    async def _boom(
        *args: object, **kwargs: object
    ) -> Result[AnnotatedDocument, Exception]:
        _ = args, kwargs
        return Failure(RuntimeError("boom"))

    monkeypatch.setattr(langextract_mod, "_extract_one", _boom)
    result = asyncio.run(
        langextract_mod._predict_one(
            "input",
            system_prompt="sys",
            examples=[],
            model_config=model_config,
            fields=("drug",),
            output_schema=output_schema,
            seed=None,
        )
    )

    assert result is None


def test_extract_predictor_predicts_and_marks_invalid(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from langextract.data import AnnotatedDocument, Extraction

    import eh30pt.cli.scripts.langextract as langextract_mod

    doc = AnnotatedDocument(
        text="input",
        extractions=[Extraction(extraction_class="drug", extraction_text="x")],
    )
    model_config = langextract_mod.build_model_config("m", "http://x", "k")
    output_schema = cast("JsonSchema", {"type": "object"})

    async def _ok(
        *args: object, **kwargs: object
    ) -> Result[AnnotatedDocument, Exception]:
        _ = args, kwargs
        return Success(doc)

    monkeypatch.setattr(langextract_mod, "_extract_one", _ok)
    predictor = langextract_mod.ExtractPredictor(
        system_prompt="sys",
        examples=[],
        model_config=model_config,
        fields=("drug",),
        output_schema=output_schema,
        seed=None,
    )
    result = asyncio.run(predictor.predict(FewShot(user="hi", label="")))

    assert result is not None

    async def _boom(
        *args: object, **kwargs: object
    ) -> Result[AnnotatedDocument, Exception]:
        _ = args, kwargs
        return Failure(RuntimeError("boom"))

    monkeypatch.setattr(langextract_mod, "_extract_one", _boom)
    with caplog.at_level("DEBUG", logger="eh30pt"):
        result = asyncio.run(predictor.predict(FewShot(user="hi", label="")))

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

    _ = mlflow_client
    mlflow.set_experiment("fold-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="fold-exp"
    )
    with RunContext(settings, config=run_config) as ctx:
        accumulator = StreamingClassificationMetrics()
        window = [
            FewShot(user="a", label="x"),
            FewShot(user="", label="y"),
            FewShot(user="c", label="y"),
        ]

        n_seen, invalid = _fold_eval_window(
            ctx,
            accumulator,
            window,
            ["x", None, "z"],
            log_interval=2,
            n_seen=0,
            invalid=0,
        )

        assert (n_seen, invalid) == (3, 1)
        assert accumulator.finalize()["accuracy"] == pytest.approx(0.5)

        logged, _ = _fold_eval_window(
            ctx,
            StreamingClassificationMetrics(),
            [FewShot(user="a", label="x"), FewShot(user="b", label="y")],
            ["x", "y"],
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

    _ = mlflow_client
    mlflow.set_experiment("stream-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="stream-exp"
    )
    samples = [
        FewShot(user="a", label="x"),
        FewShot(user="", label="y"),
        FewShot(user="c", label="y"),
    ]
    with RunContext(settings, config=run_config) as ctx:
        bar = tqdm(desc="evaluating", unit="samples", file=io.StringIO())
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


def test_export_writes_label_studio_tasks(
    mlflow_client: MlflowClient, run_config: RunConfig, tmp_path: Path
) -> None:
    from eh30pt.cli.scripts.export import _locate_span
    from eh30pt.cli.scripts.export import main as export_main

    _ = mlflow_client
    mlflow.set_experiment("export-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="export-exp"
    )
    output = tmp_path / "out" / "tasks.json"
    user = "Patsient katkestas Nebileti, vererõhk normaliseerus."
    phrase = "katkestas Nebileti"
    with RunContext(settings, config=run_config) as ctx:
        run_id = ctx.run_id
        batch = [
            GenerateRecord(
                id=0,
                user=user,
                prediction=json.dumps(
                    {
                        "phrase": phrase,
                        "reason": "vererõhk normaliseerus",
                        "drug": "Nebilet",
                    }
                ),
            ),
            GenerateRecord(
                id=1,
                user="kaebusi ei ole",
                prediction=json.dumps({"phrase": "", "reason": "", "drug": ""}),
            ),
            GenerateRecord(
                id=2,
                user="Teine tekst katkestas Nebileti.",
                prediction=json.dumps({"phrase": phrase, "reason": "x", "drug": "y"}),
            ),
            GenerateRecord(
                id=3,
                user="Arst jättis statiini ära.",
                prediction=json.dumps(
                    {
                        "phrase": "Arst jättis statiini ära.",
                        "reason": "",
                        "drug": "statiin",
                    }
                ),
            ),
            GenerateRecord(
                id=4,
                user="Täiesti tavaline tekst.",
                prediction=json.dumps(
                    {"phrase": "olematu fraas", "reason": "r", "drug": "d"}
                ),
            ),
        ]
        log_shard(ctx, tmp_path, MlflowSettings.predictions_path, 0, batch)

    export_main(
        settings=MlflowSettings(tracking_uri=mlflow_client.tracking_uri, run_id=run_id),
        output=output,
    )

    assert output.is_file()
    tasks = json.loads(output.read_text(encoding="utf-8"))

    assert len(tasks) == 1
    assert tasks[0]["data"] == {
        "text": user,
        "phrase": phrase,
        "reason": "vererõhk normaliseerus",
        "drug": "Nebilet",
    }

    start, end = user.index(phrase), user.index(phrase) + len(phrase)
    drug_start = user.index("Nebilet")
    assert _locate_span(user, phrase) == (start, end)
    assert _locate_span(user, "") is None
    assert _locate_span(user, "puudub") is None
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
                        "start": drug_start,
                        "end": drug_start + len("Nebilet"),
                        "text": "Nebilet",
                        "labels": ["drug"],
                    },
                },
            ],
            "model_version": run_id,
        }
    ]

    config_path = output.with_suffix(".xml")
    assert config_path.is_file()

    config = config_path.read_text(encoding="utf-8")

    assert '<Text name="text" value="$text" />' in config
    assert '<View style="display: none">' in config
    assert '<Labels name="span" toName="text">' in config
    assert '<Label value="phrase" />' not in config
    assert '<Label value="drug" />' not in config
    assert 'value="phrase" background="#ffe08a"' in config
    assert 'value="drug" background="#a8e6a3"' in config
    assert config.index("<Choices ") < config.index('<View style="display: none">')

    assert '<Header value="Esimese taseme märgendid" />' in config
    assert '<Header value="Drug discontinuation alamliigid" />' in config
    assert '<Header value="Metamärgendused fraasi kvaliteedi kohta" />' in config

    assert (
        '<Choices name="label" toName="text" choice="multiple" showInLine="true">'
        in config
    )
    assert (
        '<Choices name="dd_subtype" toName="text" choice="multiple" showInLine="true">'
        in config
    )
    assert (
        '<Choices name="quality" toName="text" choice="multiple" showInLine="true">'
        in config
    )

    first_level = (
        "Adverse reactions (ADR)",
        "Drug discontinuation (DD)",
        "Dosage change (DC)",
        "Incorrect phrase",
    )
    subtypes = (
        "Treatment success",
        "Treatment inefficacy",
        "Contraindication",
    )
    quality = (
        "Incomplete phrase",
        "Complex fact",
        "Incorrect phrase",
    )
    for label in (*first_level, *subtypes, *quality):
        assert f'value="{label}"' in config
        assert f'value="{label}" hint=' in config

    for label in ("Non-medical reasons", "Other", "Indeterminate"):
        assert f'value="{label}"' not in config


def test_export_reads_struct_predictions(
    mlflow_client: MlflowClient, run_config: RunConfig, tmp_path: Path
) -> None:
    from eh30pt.cli.scripts.export import main as export_main

    _ = mlflow_client
    mlflow.set_experiment("export-struct-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="export-struct-exp"
    )
    output = tmp_path / "out" / "tasks.json"
    user = "Tekkis köha, enalapriil ära jäeti."
    phrase = "enalapriil"
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
                    "prediction": {
                        "phrase": phrase,
                        "reason": "tekkis köha",
                        "drug": "enalapriil",
                    },
                },
                {
                    "id": 1,
                    "user": "kaebusi ei ole",
                    "prediction": {"phrase": "", "reason": "", "drug": ""},
                },
            ],
        )

    export_main(
        settings=MlflowSettings(tracking_uri=mlflow_client.tracking_uri, run_id=run_id),
        output=output,
    )

    tasks = json.loads(output.read_text(encoding="utf-8"))

    assert len(tasks) == 1
    assert tasks[0]["data"]["text"] == user

    start = user.index(phrase)
    span = tasks[0]["predictions"][0]["result"][0]["value"]
    assert (span["start"], span["end"]) == (start, start + len(phrase))
    assert tasks[0]["predictions"][0]["model_version"] == run_id


def test_export_rejects_non_extraction_predictions(
    mlflow_client: MlflowClient, run_config: RunConfig, tmp_path: Path
) -> None:
    from eh30pt.cli.scripts.export import main as export_main

    _ = mlflow_client
    mlflow.set_experiment("export-plain-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="export-plain-exp"
    )
    with RunContext(settings, config=run_config) as ctx:
        run_id = ctx.run_id
        batch = [GenerateRecord(id=0, user="a", prediction="Adverse reactions")]
        log_shard(ctx, tmp_path, MlflowSettings.predictions_path, 0, batch)

    with pytest.raises(ValueError, match="not extraction JSON"):
        export_main(
            settings=MlflowSettings(
                tracking_uri=mlflow_client.tracking_uri, run_id=run_id
            ),
            output=tmp_path / "out.json",
        )


def test_export_rejects_non_object_predictions(
    mlflow_client: MlflowClient, run_config: RunConfig, tmp_path: Path
) -> None:
    from eh30pt.cli.scripts.export import main as export_main

    _ = mlflow_client
    mlflow.set_experiment("export-number-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="export-number-exp"
    )
    with RunContext(settings, config=run_config) as ctx:
        run_id = ctx.run_id
        batch = [GenerateRecord(id=0, user="a", prediction="123")]
        log_shard(ctx, tmp_path, MlflowSettings.predictions_path, 0, batch)

    with pytest.raises(ValueError, match="not extraction JSON"):
        export_main(
            settings=MlflowSettings(
                tracking_uri=mlflow_client.tracking_uri, run_id=run_id
            ),
            output=tmp_path / "out.json",
        )


def test_export_writes_empty_list_without_shards(
    mlflow_client: MlflowClient, run_config: RunConfig, tmp_path: Path
) -> None:
    from eh30pt.cli.scripts.export import main as export_main

    _ = mlflow_client
    mlflow.set_experiment("export-empty-exp")
    settings = MlflowSettings(
        tracking_uri=mlflow_client.tracking_uri, experiment="export-empty-exp"
    )
    output = tmp_path / "out" / "tasks.json"
    with RunContext(settings, config=run_config) as ctx:
        run_id = ctx.run_id

    export_main(
        settings=MlflowSettings(tracking_uri=mlflow_client.tracking_uri, run_id=run_id),
        output=output,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == []


def test_export_without_run_id_raises(tmp_path: Path) -> None:
    from eh30pt.cli.scripts.export import main as export_main

    with pytest.raises(ValueError, match="no mlflow run id"):
        export_main(
            settings=MlflowSettings(tracking_uri="sqlite:///mlflow.db"),
            output=tmp_path / "out.csv",
        )


def test_model_raises_when_server_serves_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from eh30pt.openai.complete import CompletionClient
    from eh30pt.settings import ServerSettings

    async def _empty() -> SimpleNamespace:
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

    from eh30pt.openai.complete import CompletionClient
    from eh30pt.settings import ServerSettings

    calls = 0

    async def _page() -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(data=[SimpleNamespace(id="org/llama")])

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
                source="trigger",
                n_shot=1,
                stream=StreamSettings(),
                mlflow=MlflowSettings(),
                seed=None,
            )
        )
