"""Label Studio round-trip options."""

from typing import Annotated

from typer import Option

LabelStudioUrlOption = Annotated[
    str,
    Option(
        "--label-studio-url",
        envvar="LABEL_STUDIO_URL",
        help="Base URL of the Label Studio server",
    ),
]

ProjectTitleOption = Annotated[
    str | None,
    Option(
        "--title",
        help="Project title, inferred from the dataset when omitted",
    ),
]

ProjectIdsOption = Annotated[
    list[int],
    Option("--project", "-p", min=1, help="Label Studio project ids to read"),
]

OptionalProjectIdsOption = Annotated[
    list[int] | None,
    Option("--project", "-p", help="Label Studio project ids training the run"),
]
