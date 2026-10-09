"""An open MLflow run plus its local checkpoint directory."""

import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, Self, TextIO, cast

import mlflow
import mlflow.transformers
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from transformers.trainer_utils import get_last_checkpoint

from eh30pt.constants import LOGDIR
from eh30pt.logging import _UvFormatter, logger
from eh30pt.modeling.arch import get_pipeline_task
from eh30pt.modeling.trainer import get_trainer
from eh30pt.settings import MlflowSettings
from eh30pt.tracking.datasets import dataset_input
from eh30pt.tracking.resume import (
    ConfigMismatchError,
    _get_experiment_id,
    _identity_view,
    _verify_config,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType

    from mlflow.entities.model_registry.prompt_version import PromptVersion
    from peft import PeftModel
    from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerFast

    from eh30pt.settings import TrainingSettings
    from eh30pt.types import (
        Architecture,
        ComputeMetricsFn,
        DatasetName,
        DataSplits,
        PipelineTask,
        RunConfig,
    )


_from_id_tag = "eh30pt.from_id"


class _RunLog(NamedTuple):
    """File-backed log capture mirroring the console without color."""

    handler: logging.Handler
    file: TextIO


class RunContext:
    """An open MLflow run plus its local checkpoint directory."""

    _client: MlflowClient
    _run_log: _RunLog

    def __init__(
        self,
        settings: MlflowSettings,
        config: RunConfig,
    ) -> None:
        """Capture the open request; the run opens on __enter__."""
        self.settings = settings
        self.config = config
        self.resumed = False

    def __enter__(self) -> Self:
        """Open a fresh run, or reattach to run_id after verifying its config."""
        uri = self.settings.tracking_uri
        exp = self.settings.experiment
        run = self.settings.run_id

        client = MlflowClient(uri)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(exp)

        if run is not None:
            experiment_id = _get_experiment_id(client, exp)
            stored = client.get_run(run)
            if stored.info.experiment_id != experiment_id:
                msg = f"run '{run}' is not in experiment '{exp}'"
                raise ConfigMismatchError(msg)

            _verify_config(client, run, self.config)
            mlflow.start_run(run_id=run)
            self.resumed = True
        else:
            mlflow.start_run(run_name=self.settings.run_name)

        active = mlflow.active_run()
        if active is None:
            msg = "no active mlflow run after start_run"
            raise RuntimeError(msg)

        self._client = client
        if self.settings.run_id is None:
            self.settings.run_id = active.info.run_id

        run_name = active.info.run_name or self.settings.run_name
        if run_name is None:
            msg = "no run name after starting run"
            raise RuntimeError(msg)

        if self.resumed:
            self._restore_run_log()
            logger.info("reattached to run '%s'", self.run_id)

        self._attach_run_log()
        try:
            if not self.resumed:
                logger.info("opened mlflow run '%s'", self.run_id)
                params = _identity_view(self.config).items()
                self.log_params({key: str(value) for key, value in params})
                self.log_dataset_input(self.config["dataset"])
                mlflow.log_dict(dict(self.config), "config.json")
        except BaseException:
            self._fail()
            raise

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _exc_traceback: TracebackType | None,
    ) -> None:
        """Upload the run log and finish the run, marking failures."""
        if exc_type is None:
            try:
                self._persist_run_log()
                mlflow.end_run("FINISHED")
            finally:
                self._detach_run_log()
        else:
            self._fail()

    def _fail(self) -> None:
        """Upload the run log with its traceback and mark the run failed."""
        try:
            self._run_log.file.write(traceback.format_exc())
            self._persist_run_log()
            mlflow.end_run("FAILED")
        finally:
            self._detach_run_log()

    def _detach_run_log(self) -> None:
        """Detach the run log capture, closing its handler and file."""
        logger.removeHandler(self._run_log.handler)
        self._run_log.handler.close()
        self._run_log.file.close()

    def _attach_run_log(self) -> None:
        """Capture the project log to the run directory, mirroring the console."""
        path = self.dir / "run.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        file = path.open("a", encoding="utf-8")
        handler: logging.Handler = logging.StreamHandler(cast("TextIO", file))
        handler.setFormatter(_UvFormatter(use_color=False))
        logger.addHandler(handler)
        self._run_log = _RunLog(handler=handler, file=file)

    def _persist_run_log(self) -> None:
        """Upload the run log as an artifact without ever raising."""
        self._run_log.file.flush()
        try:
            mlflow.log_artifact(self._run_log.file.name, artifact_path="logs")
        except Exception:  # noqa: BLE001
            logger.warning("could not upload run log for run '%s'", self.run_id)

    def _restore_run_log(self) -> None:
        """Download a previously uploaded run log so continuation appends to it."""
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            self.download_artifacts("logs/run.log", str(self.dir))
        except MlflowException:
            logger.debug("no previous run log for run '%s'", self.run_id)

    @property
    def run_id(self) -> str:
        """Return the id of the open run."""
        if self.settings.run_id is None:
            msg = "RunContext must be opened with __enter__ before use"
            raise RuntimeError(msg)

        return self.settings.run_id

    @property
    def dir(self) -> Path:
        """Return the local checkpoint directory for this run."""
        base = LOGDIR if LOGDIR.is_absolute() else Path.cwd() / LOGDIR
        return base / self.run_id

    @property
    def artifact_uri(self) -> str:
        """Return the artifact location of this run."""
        uri = self._client.get_run(self.run_id).info.artifact_uri
        if uri is None:
            msg = f"run '{self.run_id}' has no artifact location"
            raise RuntimeError(msg)

        return uri

    def record_from_id(self, from_id: int) -> None:
        """Append from_id to the run tag tracking which id ranges were streamed."""
        previous = self._client.get_run(self.run_id).data.tags.get(_from_id_tag)
        history = f"{previous},{from_id}" if previous else str(from_id)
        self._client.set_tag(self.run_id, _from_id_tag, history)

    def next_shard_index(self, prefix: str = MlflowSettings.predictions_path) -> int:
        """Return one past the highest shard index already logged under prefix."""
        try:
            infos = self._client.list_artifacts(self.run_id, prefix)
        except MlflowException:
            logger.debug("no artifacts under '%s' for run '%s'", prefix, self.run_id)
            return 0

        indices: list[int] = []
        for info in infos:
            if info.is_dir:
                continue

            name = Path(info.path).name
            if not name.startswith(f"{prefix}-") or not name.endswith(".parquet"):
                continue

            stem = name[len(prefix) + 1 : -len(".parquet")]
            if stem.isdigit():
                indices.append(int(stem))

        return max(indices, default=-1) + 1

    def existing_max_step(self) -> int:
        """Return the highest metric step already logged on the run."""
        run = self._client.get_run(self.run_id)
        steps = [0]
        for key in run.data.metrics:
            steps.extend(
                metric.step
                for metric in self._client.get_metric_history(self.run_id, key)
            )

        return max(steps)

    def log_params(self, params: Mapping[str, str]) -> None:
        """Log run params."""
        mlflow.log_params(dict(params))

    def log_metrics(self, metrics: Mapping[str, float], *, step: int = 0) -> None:
        """Log run metrics at step."""
        mlflow.log_metrics(dict(metrics), step=step)

    def log_dataset_input(self, dataset: DatasetName) -> None:
        """Log dataset as a run input so it shows in the datasets tab."""
        mlflow.log_input(dataset_input(dataset))

    def sync_checkpoints(self, artifact_path: str = "checkpoints") -> None:
        """Log local HF checkpoints as run artifacts, skipping when none exist."""
        names = self._checkpoint_names()
        if not names:
            logger.debug("no checkpoints in '%s', skipping artifact sync", self.dir)
            return

        for name in names:
            mlflow.log_artifacts(
                str(self.dir / name),
                artifact_path=f"{artifact_path}/{name}",
            )

        logger.info("synced %d checkpoints of run '%s'", len(names), self.run_id)

    def _checkpoint_names(self) -> list[str]:
        """Return HF checkpoint dir names under output dir, following its naming."""
        if not self.dir.is_dir():
            return []

        return sorted(
            child.name
            for child in self.dir.iterdir()
            if child.is_dir() and child.name.startswith("checkpoint-")
        )

    def find_checkpoint(self, artifact_path: str = "checkpoints") -> str | None:
        """Find the latest HF checkpoint, restoring it from artifacts if needed."""
        if self.dir.is_dir():
            checkpoint = get_last_checkpoint(str(self.dir))
            if checkpoint is not None:
                logger.info("resuming from checkpoint '%s'", checkpoint)
                return checkpoint

        if self._restore_checkpoints(artifact_path):
            checkpoint = get_last_checkpoint(str(self.dir))
            if checkpoint is not None:
                logger.info("resuming from restored checkpoint '%s'", checkpoint)
                return checkpoint

        logger.info("no checkpoint for run '%s', training from scratch", self.run_id)
        return None

    def _restore_checkpoints(self, artifact_path: str) -> bool:
        """Download checkpoint artifacts into the run directory."""
        try:
            infos = self._client.list_artifacts(self.run_id, artifact_path)
        except MlflowException:
            logger.debug("no checkpoint artifacts for run '%s'", self.run_id)
            return False

        names = [Path(info.path).name for info in infos if info.is_dir]
        if not names:
            return False

        self.dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            self._client.download_artifacts(
                self.run_id,
                f"{artifact_path}/{name}",
                str(self.dir),
            )

        logger.info("restored %d checkpoints of run '%s'", len(names), self.run_id)
        return True

    def log_model_artifacts(
        self,
        model: PreTrainedModel | PeftModel,
        tokenizer: PreTrainedTokenizerFast,
        task: PipelineTask,
        name: str = MlflowSettings.model_path,
    ) -> None:
        """Log model and tokenizer as a flavored model so the run shows it."""
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            task=task,
            name=name,
            pip_requirements=["accelerate", "peft", "torch", "transformers"],
        )

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """Log a local file as a run artifact."""
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def download_artifacts(self, artifact_path: str, dst_path: str) -> str:
        """Download run artifacts, raising MlflowException when missing."""
        return self._client.download_artifacts(self.run_id, artifact_path, dst_path)

    def register_prompt_version(
        self,
        name: str,
        template: str,
        commit_message: str,
    ) -> PromptVersion:
        """Commit template under name unless the latest version matches."""
        try:
            versions = self._client.search_prompt_versions(name)
        except MlflowException:
            logger.debug("no prompt named '%s' yet", name)
            versions = []

        if versions and versions[0].template == template:
            prompt_version = versions[0]
            logger.debug(
                "prompt '%s' unchanged at version %s",
                name,
                versions[0].version,
            )
        else:
            logger.info("commiting prompt for '%s'", name)
            prompt_version = self._client.register_prompt(
                name=name,
                template=template,
                commit_message=commit_message,
            )

        self._client.link_prompt_version_to_run(self.run_id, prompt_version)
        return prompt_version

    def register_best_model(
        self,
        name: str,
        test_metrics: Mapping[str, float],
        metric: str = "test_f1",
        mode: Literal["max", "min"] = "max",
    ) -> int | None:
        """Register the run's model only when it beats the best alias."""
        try:
            self._client.get_registered_model(name)
        except MlflowException:
            logger.info("creating registered model '%s'", name)
            self._client.create_registered_model(name)

        if metric not in test_metrics:
            logger.warning(
                "metric '%s' not in test metrics, skipping registration",
                metric,
            )

            return None

        current_best: float | None = None
        try:
            current = self._client.get_model_version_by_alias(name, "best")
            if metric in current.tags:
                current_best = float(current.tags[metric])
        except MlflowException:
            logger.debug("no best version of '%s' yet", name)

        candidate = test_metrics[metric]
        beaten = current_best is None or (
            candidate > current_best if mode == "max" else candidate < current_best
        )

        if not beaten:
            logger.info(
                "not registering '%s': %s=%f does not beat %f",
                name,
                metric,
                candidate,
                current_best,
            )

            return None

        raw_version = self._client.create_model_version(
            name,
            source=f"runs:/{self.run_id}/{MlflowSettings.model_path}",
            run_id=self.run_id,
        ).version

        version = cast("int", raw_version)

        for key, value in test_metrics.items():
            self._client.set_model_version_tag(name, str(version), key, str(value))

        logger.info(
            "new best version %s of '%s' (%s=%f)",
            version,
            name,
            metric,
            test_metrics[metric],
        )

        self._client.set_registered_model_alias(name, "best", str(version))
        return version

    def train_and_evaluate(
        self,
        *,
        model: PreTrainedModel | PeftModel,
        tokenizer: PreTrainedTokenizerFast,
        data: DataSplits,
        arch: Architecture,
        collate_fn: DataCollator,
        metrics_fn: ComputeMetricsFn,
        registered_name: str,
        training: TrainingSettings,
    ) -> dict[str, float]:
        """Train model under run context, then evaluate, log, and register the best."""
        run = self.settings.run_name
        if run is None:
            msg = "no mlflow run name"
            raise ValueError(msg)

        trainer = get_trainer(
            model=model,
            tokenizer=tokenizer,
            data=data,
            arch=arch,
            collate_fn=collate_fn,
            metrics_fn=metrics_fn,
            output_dir=self.dir,
            eval_steps=training.eval_steps,
            patience=training.patience,
            tolerance=training.tolerance,
            train_steps=training.train_steps,
            learning_rate=training.learning_rate,
            batch_size=training.batch_size,
            warmup_ratio=training.warmup_ratio,
            run_name=run,
            report_to="mlflow",
            checkpoints=True,
        )

        logger.debug("starting trainer")
        trainer.train(resume_from_checkpoint=self.find_checkpoint())
        self.sync_checkpoints()

        logger.debug("starting test eval")
        test_metrics = trainer.evaluate(data.test, metric_key_prefix="test")

        self.finish_training(
            registered_name,
            model=model,
            tokenizer=tokenizer,
            task=get_pipeline_task(arch),
            test_metrics=dict(test_metrics),
        )

        return test_metrics

    def evaluate_and_log(
        self,
        *,
        model: PreTrainedModel | PeftModel,
        tokenizer: PreTrainedTokenizerFast,
        data: DataSplits,
        arch: Architecture,
        collate_fn: DataCollator,
        metrics_fn: ComputeMetricsFn,
        batch_size: int,
    ) -> dict[str, float]:
        """Evaluate model on the test split and log the returned metrics."""
        run = self.settings.run_name
        if run is None:
            msg = "no mlflow run name"
            raise ValueError(msg)

        trainer = get_trainer(
            model=model,
            tokenizer=tokenizer,
            data=data,
            arch=arch,
            collate_fn=collate_fn,
            metrics_fn=metrics_fn,
            output_dir=self.dir,
            train_steps=1,
            eval_steps=0,
            batch_size=batch_size,
            run_name=run,
            report_to="mlflow",
        )

        logger.debug("starting test eval")
        test_metrics = trainer.evaluate(data.test, metric_key_prefix="test")
        self.log_metrics(dict(test_metrics))
        return test_metrics

    def finish_training(
        self,
        registered_name: str,
        *,
        model: PreTrainedModel | PeftModel,
        tokenizer: PreTrainedTokenizerFast,
        task: PipelineTask,
        test_metrics: Mapping[str, float],
        metric: str = "test_f1",
        mode: Literal["max", "min"] = "max",
    ) -> None:
        """Log test metrics and artifacts, registering the model when it leads."""
        self.log_metrics(dict(test_metrics))
        self.log_model_artifacts(model, tokenizer, task)
        self.register_best_model(registered_name, test_metrics, metric, mode)

    def log_prompt_version(
        self,
        dataset: str,
        arch: Architecture,
        system_prompt: str,
    ) -> None:
        """Register the system prompt, recording its version on fresh runs."""
        run = self.settings.run_name
        if run is None:
            msg = "no mlflow run name"
            raise ValueError(msg)

        prompt_version = self.register_prompt_version(
            f"{dataset}-{arch}",
            system_prompt,
            commit_message=f"tracked from run {run}",
        )

        if not self.resumed:
            self.log_params({"prompt_version": str(prompt_version.version)})
