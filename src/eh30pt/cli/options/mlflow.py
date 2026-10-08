"""MLflow tracking options."""

from typing import Annotated

from typer import Option

TrackingUriOption = Annotated[
    str,
    Option(envvar="MLFLOW_TRACKING_URI", help="MLflow tracking URI"),
]

ExperimentOption = Annotated[
    str,
    Option(envvar="MLFLOW_EXPERIMENT", help="Experiment name for mlflow tracking"),
]

ExportRunOption = Annotated[
    str,
    Option(
        "--mlflow-run-id",
        envvar="MLFLOW_RUN_ID",
        help="Run whose prediction shards to export",
    ),
]

RunNameOption = Annotated[
    str | None,
    Option(
        envvar="MLFLOW_RUN_NAME",
        help="Run name for mlflow tracking, inferred from parameters by default",
    ),
]

RunIdOption = Annotated[
    str | None,
    Option(
        "--mlflow-run-id",
        envvar="MLFLOW_RUN_ID",
        help="Reattach to an existing run and append results",
    ),
]
