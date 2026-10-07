"""Label Studio API client for creating projects and importing tasks."""

from typing import TYPE_CHECKING, NamedTuple, Self, TypedDict, cast

import httpx
from returns.result import safe

from eh30pt.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from httpx._types import QueryParamTypes

    from eh30pt.labeling.types import Task
    from eh30pt.settings import LabelStudioSettings


class ProjectRef(NamedTuple):
    """Identity of a Label Studio project."""

    id: int
    title: str


class ImportedTasks(NamedTuple):
    """Counts reported by the server after an import."""

    tasks: int
    predictions: int


class StudioTask(TypedDict):
    """One imported task as the server reports it."""

    id: int
    inner_id: int
    data: dict[str, str]


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
            body = _object(
                self._request(
                    "GET",
                    "/api/tasks/",
                    params={
                        "project": project,
                        "page": page,
                        "page_size": page_size,
                        "fields": "all",
                    },
                ),
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
        body = _object(
            self._request(
                "POST",
                "/api/projects/",
                payload={"title": title, "label_config": label_config},
            ),
            "the created project",
        )
        return ProjectRef(id=_number(body, "id", "the created project"), title=title)

    @safe
    def validate(self, project: int, label_config: str) -> None:
        """Ask the server to parse the labeling config, failing when invalid."""
        logger.debug("validating the labeling config of project %d", project)
        self._request(
            "POST",
            f"/api/projects/{project}/validate",
            payload={"label_config": label_config},
        )

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
                self._request(
                    "POST",
                    f"/api/projects/{project}/import",
                    payload=chunk,
                ),
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
        """Collect every task of a project."""
        return [
            StudioTask(
                id=_number(page, "id", "a task"),
                inner_id=_number(page, "inner_id", "a task"),
                data=cast("dict[str, str]", page.get("data", {})),
            )
            for page in self._task_pages(project, page_size=page_size)
        ]
