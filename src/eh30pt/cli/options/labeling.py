"""Label Studio round-trip options."""

import rich_click as click

label_studio_url_option = click.option(
    "--label-studio-url",
    "label_studio_url",
    type=str,
    default="http://localhost:8080",
    show_default=True,
    show_envvar=True,
    envvar="LABEL_STUDIO_URL",
    help="Base URL of the Label Studio server",
)

project_title_option = click.option(
    "--title",
    "title",
    type=str,
    default=None,
    help="Project title, inferred from the dataset when omitted",
)

project_ids_option = click.option(
    "--project",
    "-p",
    "labeling_projects",
    type=click.IntRange(min=1),
    required=True,
    multiple=True,
    help="Label Studio project ids to read",
)

optional_project_ids_option = click.option(
    "--project",
    "-p",
    "labeling_projects",
    type=int,
    default=None,
    multiple=True,
    help="Label Studio project ids training the run",
)
