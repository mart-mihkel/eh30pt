"""An open MLflow run plus its local checkpoint directory."""

from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException
from transformers.trainer_utils import get_last_checkpoint

from eh30pt.constants import LOGDIR, MLFLOW_ARTIFACT_PATH
from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from mlflow.entities.model_registry.prompt_version import PromptVersion
    from mlflow.tracking import MlflowClient
    from peft import PeftModel
    from transformers import PreTrainedModel, PreTrainedTokenizerFast


def _checkpoint_dir(run_id: str) -> Path:
    """Return the local checkpoint directory for a run id."""
    return LOGDIR / run_id


def _find_checkpoint(output_dir: Path | str) -> str | None:
    """Find the latest HF checkpoint in `output_dir`, if any exists."""
    checkpoint = get_last_checkpoint(str(output_dir))
    if checkpoint is None:
        logger.info("no checkpoint in '%s', training from scratch", output_dir)
        return None

    logger.info("resuming from checkpoint '%s'", checkpoint)
    return checkpoint


@dataclass
class RunContext:
    """
    An open MLflow run plus its local checkpoint directory.

    All tracking calls go through the client. The fluent mlflow module is
    only used to keep an active run for the HF Trainer autologging callback.

    Attributes
    ----------
    client : MlflowClient
        MLflow client bound to the run's tracking URI.
    experiment : str
        Name of the experiment owning the run.
    run_id : str
        MLflow run id; also keys the local checkpoint directory.
    run_name : str | None
        Human-readable display name.
    tracking_uri : str
        URI of the MLflow tracking server.
    resumed : bool
        Whether the run was reattached to a previous unfinished run.

    """

    client: MlflowClient
    experiment: str
    run_id: str
    run_name: str | None
    tracking_uri: str
    resumed: bool

    @property
    def dir(self) -> Path:
        """Return the local checkpoint directory for this run."""
        return _checkpoint_dir(self.run_id)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """
        Log run params, stringifying values.

        Parameters
        ----------
        params : Mapping[str, Any]
            Param names to values; values are stored via `str`.

        """
        self.client.log_batch(
            self.run_id,
            params=[Param(key, str(value)) for key, value in params.items()],
        )

    def log_metrics(self, metrics: Mapping[str, float], *, step: int = 0) -> None:
        """
        Log run metrics at step.

        Parameters
        ----------
        metrics : Mapping[str, float]
            Metric names to values.
        step : int
            Training step the values belong to.

        """
        import time

        self.client.log_batch(
            self.run_id,
            metrics=[
                Metric(key, value, int(time.time() * 1000), step)
                for key, value in metrics.items()
            ],
        )

    def find_checkpoint(self) -> str | None:
        """Find the latest HF checkpoint of this run, if any exists."""
        return _find_checkpoint(self.dir)

    def log_model_artifacts(
        self,
        model: PreTrainedModel | PeftModel,
        tokenizer: PreTrainedTokenizerFast,
        artifact_path: str = MLFLOW_ARTIFACT_PATH,
    ) -> None:
        """
        Save model and tokenizer, logging them as run artifacts.

        Parameters
        ----------
        model : PreTrainedModel | PeftModel
            Trained model (or PEFT adapter) to save.
        tokenizer : PreTrainedTokenizerFast
            Tokenizer to save alongside the model.
        artifact_path : str
            Artifact directory inside the run.

        """
        with TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            tokenizer.save_pretrained(tmp)
            self.client.log_artifacts(self.run_id, tmp, artifact_path=artifact_path)

    def register_prompt_version(
        self,
        name: str,
        template: str,
        commit_message: str,
    ) -> PromptVersion:
        """
        Commit template under name unless the latest version matches.

        Parameters
        ----------
        name : str
            Prompt registry name.
        template : str
            Prompt template text to commit.
        commit_message : str
            Message recorded with the new version.

        Returns
        -------
        PromptVersion
            The reused or newly committed prompt version, linked to the run.

        """
        try:
            versions = self.client.search_prompt_versions(name)
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
            prompt_version = self.client.register_prompt(
                name=name,
                template=template,
                commit_message=commit_message,
            )

        self.client.link_prompt_version_to_run(self.run_id, prompt_version)
        return prompt_version

    def register_best_model(
        self,
        name: str,
        test_metrics: Mapping[str, float],
        metric: str = "test_f1",
    ) -> str | None:
        """
        Register the run's model only when it beats the best alias.

        Parameters
        ----------
        name : str
            Registered model name.
        test_metrics : Mapping[str, float]
            Test metrics to tag the version with; must contain `metric`.
        metric : str
            Metric deciding whether the run leads.

        Returns
        -------
        str | None
            The created model version number, or None when the run does not
            improve on the current best.

        """
        try:
            self.client.get_registered_model(name)
        except MlflowException:
            logger.info("creating registered model '%s'", name)
            self.client.create_registered_model(name)

        if metric not in test_metrics:
            logger.warning(
                "metric '%s' not in test metrics, skipping registration",
                metric,
            )

            return None

        current_best: float | None = None
        try:
            current = self.client.get_model_version_by_alias(name, "best")
            if metric in current.tags:
                current_best = float(current.tags[metric])
        except MlflowException:
            logger.debug("no best version of '%s' yet", name)

        if current_best is not None and test_metrics[metric] <= current_best:
            logger.info(
                "not registering '%s': %s=%f does not beat %f",
                name,
                metric,
                test_metrics[metric],
                current_best,
            )

            return None

        version = self.client.create_model_version(
            name,
            source=f"runs:/{self.run_id}/{MLFLOW_ARTIFACT_PATH}",
            run_id=self.run_id,
        ).version

        for key, value in test_metrics.items():
            self.client.set_model_version_tag(name, version, key, str(value))

        logger.info(
            "new best version %s of '%s' (%s=%f)",
            version,
            name,
            metric,
            test_metrics[metric],
        )

        self.client.set_registered_model_alias(name, "best", version)
        return version

    def finish_training(
        self,
        registered_name: str,
        *,
        model: PreTrainedModel | PeftModel,
        tokenizer: PreTrainedTokenizerFast,
        test_metrics: Mapping[str, float],
        metric: str = "test_f1",
    ) -> None:
        """
        Log test metrics and artifacts, registering the model when it leads.

        Parameters
        ----------
        registered_name : str
            Registered model name for the best-model comparison.
        model : PreTrainedModel | PeftModel
            Trained model (or PEFT adapter) to log.
        tokenizer : PreTrainedTokenizerFast
            Tokenizer to log alongside the model.
        test_metrics : Mapping[str, float]
            Test metrics to log and tag the version with.
        metric : str
            Metric deciding whether the run leads.

        """
        self.log_metrics(dict(test_metrics))
        self.log_model_artifacts(model, tokenizer)
        self.register_best_model(registered_name, test_metrics, metric)
