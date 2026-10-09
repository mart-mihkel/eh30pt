"""Label Studio round-trip options."""

import rich_click as click

# Unannotated on purpose: click decorators are generic over the decorated
# callable, which a variable annotation cannot express, so the precise type
# is left for inference.

LabelStudioUrlOption = click.option(
    "--label-studio-url",
    "label_studio_url",
    type=str,
    default="http://localhost:8080",
    show_default=True,
    show_envvar=True,
    envvar="LABEL_STUDIO_URL",
    help="Base URL of the Label Studio server",
)

ProjectTitleOption = click.option(
    "--title",
    "title",
    type=str,
    default=None,
    help="Project title, inferred from the dataset when omitted",
)

ProjectIdsOption = click.option(
    "--project",
    "-p",
    "labeling_projects",
    type=click.IntRange(min=1),
    required=True,
    multiple=True,
    help="Label Studio project ids to read",
)

OptionalProjectIdsOption = click.option(
    "--project",
    "-p",
    "labeling_projects",
    type=int,
    default=None,
    multiple=True,
    help="Label Studio project ids training the run",
)
