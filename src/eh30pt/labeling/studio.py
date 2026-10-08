"""Label Studio API client for creating projects and importing tasks."""

from typing import TYPE_CHECKING, NamedTuple, Self, TypedDict, cast

import httpx
import polars as pl
from returns.result import safe

from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from httpx._types import QueryParamTypes

    from eh30pt.labeling.types import ChoiceValue, SpanValue, Task
    from eh30pt.settings import LabelStudioSettings


class ProjectRef(NamedTuple):
    """Identity of a Label Studio project."""

    id: int
    title: str


class ImportedTasks(NamedTuple):
    """Counts reported by the server after an import."""

    tasks: int
    predictions: int


class StudioResult(TypedDict):
    """One entry of an annotation result."""

    from_name: str
    to_name: str
    type: str
    value: ChoiceValue | SpanValue


class StudioAnnotation(TypedDict):
    """One annotator submission attached to a task."""

    id: int
    result: list[StudioResult]


class StudioTask(TypedDict):
    """One imported task as the server reports it."""

    id: int
    inner_id: int
    data: dict[str, str]
    annotations: list[StudioAnnotation]


def _json(response: httpx.Response, what: str) -> object:
    """
    Return the JSON body of a successful response.

    Label Studio answers a delete with 204 and no body, so a success without
    content is reported as an empty object rather than parsed.
    """
    if not response.is_success:
        body = response.text[:200]
        msg = f"label studio refused to {what}: {response.status_code} {body}"
        raise RuntimeError(msg)

    if response.status_code == httpx.codes.NO_CONTENT or not response.content:
        return {}

    return response.json()


def _object(payload: object, what: str) -> dict[str, object]:
    """Narrow a JSON payload to an object."""
    if not isinstance(payload, dict):
        msg = f"{what} is not a json object"
        raise TypeError(msg)

    return cast("dict[str, object]", payload)


def _number(payload: dict[str, object], key: str, what: str) -> int:
    """Read an integer field, raising when the server left it out."""
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{what} has no integer '{key}'"
        raise TypeError(msg)

    return value


def _results(payload: object, what: str) -> list[dict[str, object]]:
    """Read the items of a paginated list response."""
    page = _object(payload, what)
    results = page.get("results")
    if not isinstance(results, list):
        msg = f"{what} carries no 'results' list"
        raise TypeError(msg)

    return cast("list[dict[str, object]]", results)


def _tasks(payload: object, what: str) -> list[dict[str, object]]:
    """Read the tasks of a task list response."""
    page = _object(payload, what)
    tasks = page.get("tasks")
    if not isinstance(tasks, list):
        msg = f"{what} carries no 'tasks' list"
        raise TypeError(msg)

    return cast("list[dict[str, object]]", tasks)


def _chosen_labels(result: StudioResult) -> tuple[str, ...]:
    """Return the labels a result selected, empty for non-choice results."""
    if result["type"] != "choices":
        return ()

    value = cast("ChoiceValue", result["value"])
    return tuple(value.get("choices", ()))


def group_choices(
    annotation: StudioAnnotation, controls: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Group one annotation's chosen labels by control, in control order."""
    chosen: dict[str, list[str]] = {control: [] for control in controls}
    for result in annotation["result"]:
        labels = chosen.get(result["from_name"])
        if labels is None:
            continue

        labels.extend(_chosen_labels(result))

    return {control: tuple(chosen[control]) for control in controls}


def annotation_frame(
    tasks: Sequence[StudioTask], controls: Sequence[str]
) -> pl.DataFrame:
    """Build one row per annotation with a label list column per control."""
    rows: list[int] = []
    annotations: list[int] = []
    texts: list[str] = []
    phrases: list[str] = []
    drugs: list[str] = []
    reasons: list[str] = []
    grouped: list[list[tuple[str, ...]]] = []

    for task in tasks:
        data = task["data"]
        for annotation in task["annotations"]:
            chosen = group_choices(annotation, controls)

            rows.append(task["id"])
            annotations.append(annotation["id"])
            texts.append(data.get("text", ""))
            phrases.append(data.get("phrase", ""))
            drugs.append(data.get("drug", ""))
            reasons.append(data.get("reason", ""))
            grouped.append([chosen[control] for control in controls])

    frame = pl.DataFrame(
        {
            "task": pl.Series("task", rows, dtype=pl.Int64),
            "annotation": pl.Series("annotation", annotations, dtype=pl.Int64),
            "text": pl.Series("text", texts, dtype=pl.String),
            "phrase": pl.Series("phrase", phrases, dtype=pl.String),
            "drug": pl.Series("drug", drugs, dtype=pl.String),
            "reason": pl.Series("reason", reasons, dtype=pl.String),
        },
    )

    for column, control in enumerate(controls):
        labels = pl.Series(
            control,
            [row[column] for row in grouped],
            dtype=pl.List(pl.String),
        )

        frame = frame.with_columns(labels)

    return frame


class StudioClient:
    """Label Studio client authenticating with a personal access token."""

    def __init__(self, settings: LabelStudioSettings) -> None:
        """Store the endpoint and the token, deferring the access exchange."""
        self._settings = settings
        self._client = httpx.Client(base_url=settings.url, timeout=60.0)
        self._access: str | None = None

    def __enter__(self) -> Self:
        """Enter the client context."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the underlying connection pool."""
        self._client.close()

    def _token(self) -> str:
        """
        Return an access token, exchanging the personal token for one.

        Label Studio issues personal access tokens as refresh tokens, so every
        call needs a short-lived access token exchanged through a second
        request.
        """
        if self._access is not None:
            return self._access

        response = self._client.post(
            "/api/token/refresh",
            json={"refresh": self._settings.token.get_secret_value()},
        )

        payload = _object(_json(response, "refresh the access token"), "the response")
        access = payload.get("access")
        if not isinstance(access, str):
            msg = "the refresh response carries no 'access' token"
            raise TypeError(msg)

        logger.debug("exchanged an access token at '%s'", self._settings.url)
        self._access = access
        return access

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: QueryParamTypes | None = None,
        payload: object | None = None,
    ) -> object:
        """Send one authorized request and return its parsed json body."""
        response = self._client.request(
            method,
            path,
            params=params,
            json=payload,
            headers={"Authorization": f"Bearer {self._token()}"},
        )

        return _json(response, f"{method.lower()} '{path}'")

    def _pages(self, path: str, *, page_size: int) -> Iterator[dict[str, object]]:
        """Yield every page of a 'results' paginated list endpoint."""
        url: str | None = path
        params: QueryParamTypes | None = {"page_size": page_size}
        while url is not None:
            body = _object(
                self._request("GET", url, params=params),
                "a project page",
            )

            next_url = body.get("next")
            url = next_url if isinstance(next_url, str) else None
            params = None  # the next url carries its own query

            yield from _results(body, "a project page")

    def _task_pages(
        self, project: int, *, page_size: int
    ) -> Iterator[dict[str, object]]:
        """
        Yield every page of the task list, which does not use 'results'.

        The task list is paginated with page and page_size and carries its
        items under 'tasks', unlike the rest of the API.
        """
        page = 1
        while True:
            params = {
                "project": project,
                "page": page,
                "page_size": page_size,
                "fields": "all",
            }

            body = _object(
                self._request("GET", "/api/tasks/", params=params),
                "a task page",
            )

            tasks = _tasks(body, "a task page")
            if not tasks:
                return

            yield from tasks
            total = _number(body, "total", "a task page")
            if len(tasks) < page_size or page * page_size >= total:
                return

            page += 1

    @safe
    def projects(self, *, page_size: int = 100) -> list[ProjectRef]:
        """List the projects of the server."""
        pages = self._pages("/api/projects/", page_size=page_size)
        return [
            ProjectRef(
                id=_number(page, "id", "a project"),
                title=str(page.get("title", "")),
            )
            for page in pages
        ]

    @safe
    def create(self, title: str, label_config: str) -> ProjectRef:
        """Create a project holding `title` and `label_config`."""
        logger.info("creating project '%s'", title)
        payload = {"title": title, "label_config": label_config}
        body = _object(
            self._request("POST", "/api/projects/", payload=payload),
            "the created project",
        )

        return ProjectRef(id=_number(body, "id", "the created project"), title=title)

    @safe
    def validate(self, project: int, label_config: str) -> None:
        """Ask the server to parse the labeling config, failing when invalid."""
        logger.debug("validating the labeling config of project %d", project)
        payload = {"label_config": label_config}
        self._request("POST", f"/api/projects/{project}/validate", payload=payload)

    @safe
    def delete(self, project: int) -> None:
        """Delete a project and everything in it."""
        logger.info("deleting project %d", project)
        self._request("DELETE", f"/api/projects/{project}")

    @safe
    def import_tasks(
        self,
        project: int,
        items: Sequence[Task],
        *,
        chunk_size: int = 100,
    ) -> ImportedTasks:
        """Import task data with their pre-annotations in chunks."""
        tasks = 0
        predictions = 0
        for start in range(0, len(items), chunk_size):
            chunk = list(items[start : start + chunk_size])
            body = _object(
                self._request("POST", f"/api/projects/{project}/import", payload=chunk),
                "an import response",
            )

            tasks += _number(body, "task_count", "an import response")
            predictions += _number(body, "prediction_count", "an import response")

            logger.info(
                "imported %d tasks and %d pre-annotations",
                tasks,
                predictions,
            )

        return ImportedTasks(tasks=tasks, predictions=predictions)

    @safe
    def tasks(self, project: int, *, page_size: int = 100) -> list[StudioTask]:
        """Collect every task of a project with its annotations."""
        return [
            StudioTask(
                id=_number(page, "id", "a task"),
                inner_id=_number(page, "inner_id", "a task"),
                data=cast("dict[str, str]", page.get("data", {})),
                annotations=cast("list[StudioAnnotation]", page.get("annotations", [])),
            )
            for page in self._task_pages(project, page_size=page_size)
        ]

    @safe
    def annotate(self, task: int, result: Sequence[StudioResult]) -> int:
        """Attach one annotation to a task, returning its id."""
        payload = {"task": task, "result": list(result)}
        body = _object(
            self._request("POST", f"/api/tasks/{task}/annotations/", payload=payload),
            "the created annotation",
        )

        return _number(body, "id", "the created annotation")
