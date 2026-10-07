"""End to end tests for creating a Label Studio project over the API."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import polars as pl
import pytest
from returns.result import Failure

from eh30pt.labeling.extract import extraction_labeling
from eh30pt.labeling.studio import StudioClient
from eh30pt.settings import LabelStudioSettings

if TYPE_CHECKING:
    from collections.abc import Iterator

    from eh30pt.labeling.studio import ProjectRef

pytestmark = pytest.mark.slow

_PREDICTIONS = pl.DataFrame(
    {
        "user": [
            "Patsient lõpetas aspiriini kulu peavalu tõttu",
            "Vähendasime statiini doosi lihasvalu tõttu",
        ],
        "prediction": [
            (
                '{"phrase": "lõpetas aspiriini kulu",'
                ' "reason": "peavalu", "drug": "aspiriini"}'
            ),
            (
                '{"phrase": "Vähendasime statiini doosi",'
                ' "reason": "lihasvalu", "drug": "statiini"}'
            ),
        ],
    },
)


@pytest.fixture
def studio() -> LabelStudioSettings:
    """Return settings for a live server, xfailing when none is configured."""
    if os.getenv("LABEL_STUDIO_TOKEN") is None:
        pytest.xfail("LABEL_STUDIO_TOKEN is not set")

    settings = LabelStudioSettings()
    with StudioClient(settings) as client:
        if isinstance(client.projects(), Failure):
            pytest.xfail(f"label studio at '{settings.url}' is unreachable")

    return settings


@pytest.fixture
def project(studio: LabelStudioSettings) -> Iterator[ProjectRef]:
    """Create a throwaway project and delete it after the test."""
    labeling = extraction_labeling("discont-extract", _PREDICTIONS, "run-1")

    with StudioClient(studio) as client:
        created = client.create("eh30pt-e2e", labeling.config).unwrap()
        yield created

        client.delete(created.id).unwrap()


def test_create_project_imports_tasks_with_pre_annotations(
    studio: LabelStudioSettings,
    project: ProjectRef,
) -> None:
    labeling = extraction_labeling("discont-extract", _PREDICTIONS, "run-1")

    with StudioClient(studio) as client:
        client.validate(project.id, labeling.config).unwrap()
        imported = client.import_tasks(project.id, labeling.tasks).unwrap()

    assert imported.tasks == len(labeling.tasks)
    assert imported.predictions == len(labeling.tasks)

    with StudioClient(studio) as client:
        tasks = client.tasks(project.id).unwrap()

    assert len(tasks) == len(labeling.tasks)
    assert tasks[0]["data"]["text"] == _PREDICTIONS["user"][0]
    assert tasks[0]["data"]["drug"] == "aspiriini"


def test_create_task_creates_a_separate_project_per_call(
    studio: LabelStudioSettings,
) -> None:
    labeling = extraction_labeling("discont-extract", _PREDICTIONS, "run-1")
    created: list[ProjectRef] = []

    _projects = 2
    with StudioClient(studio) as client:
        try:
            for index in (1, 2):
                project = client.create(f"eh30pt-e2e-{index}", labeling.config).unwrap()
                created.append(project)

            ids = {project.id for project in created}
            assert len(ids) == _projects

            titles = {project.title for project in client.projects().unwrap()}
            assert {"eh30pt-e2e-1", "eh30pt-e2e-2"} <= titles
        finally:
            for project in created:
                client.delete(project.id).unwrap()


def test_validate_rejects_a_malformed_label_config(
    studio: LabelStudioSettings,
    project: ProjectRef,
) -> None:
    with StudioClient(studio) as client:
        assert isinstance(client.validate(project.id, "<View><Text"), Failure)


def test_delete_removes_the_project(studio: LabelStudioSettings) -> None:
    labeling = extraction_labeling("discont-extract", _PREDICTIONS, "run-1")

    with StudioClient(studio) as client:
        created = client.create("eh30pt-e2e-delete", labeling.config).unwrap()
        assert created.id in {project.id for project in client.projects().unwrap()}

        client.delete(created.id).unwrap()
        assert created.id not in {project.id for project in client.projects().unwrap()}
