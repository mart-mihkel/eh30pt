"""MLflow tracking options."""

import rich_click as click

tracking_uri_option = click.option(
    "--mlflow-tracking-uri",
    "mlflow_tracking_uri",
    type=str,
    show_default=True,
    show_envvar=True,
    envvar="MLFLOW_TRACKING_URI",
    help="MLflow tracking URI",
)

experiment_option = click.option(
    "--mlflow-experiment",
    "mlflow_experiment",
    type=str,
    default="eh30pt",
    show_default=True,
    show_envvar=True,
    envvar="MLFLOW_EXPERIMENT",
    help="Experiment name for mlflow tracking",
)

export_run_option = click.option(
    "--mlflow-run-id",
    "mlflow_run_id",
    type=str,
    required=True,
    show_envvar=True,
    envvar="MLFLOW_RUN_ID",
    help="Run whose prediction shards to export",
)

run_name_option = click.option(
    "--mlflow-run-name",
    "mlflow_run_name",
    type=str,
    default=None,
    show_envvar=True,
    envvar="MLFLOW_RUN_NAME",
    help="Run name for mlflow tracking, inferred from parameters by default",
)

run_id_option = click.option(
    "--mlflow-run-id",
    "mlflow_run_id",
    type=str,
    default=None,
    show_envvar=True,
    envvar="MLFLOW_RUN_ID",
    help="Reattach to an existing run and append results",
)
