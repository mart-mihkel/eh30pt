"""End to end tests for creating a Label Studio project over the API."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import polars as pl
import pytest
from returns.result import Failure

from eh30pt.labeling.studio import StudioClient
from eh30pt.labeling.tasks.extract import extraction_labeling
from eh30pt.settings import LabelStudioSettings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from eh30pt.labeling.studio import ProjectRef

pytestmark = pytest.mark.slow

_preds = pl.DataFrame(
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
    labeling = extraction_labeling("discont-extract/trigger", _preds, "run-1")

    with StudioClient(studio) as client:
        created = client.create("eh30pt-e2e", labeling.config).unwrap()
        yield created

        client.delete(created.id).unwrap()


def test_create_project_imports_tasks_with_pre_annotations(
    studio: LabelStudioSettings,
    project: ProjectRef,
) -> None:
    labeling = extraction_labeling("discont-extract/trigger", _preds, "run-1")

    with StudioClient(studio) as client:
        client.validate(project.id, labeling.config).unwrap()
        imported = client.import_tasks(project.id, labeling.tasks).unwrap()

    assert imported.tasks == len(labeling.tasks)
    assert imported.predictions == len(labeling.tasks)

    with StudioClient(studio) as client:
        tasks = client.tasks(project.id).unwrap()

    assert len(tasks) == len(labeling.tasks)
    assert tasks[0]["data"]["text"] == _preds["user"][0]
    assert tasks[0]["data"]["drug"] == "aspiriini"


def test_create_task_creates_a_separate_project_per_call(
    studio: LabelStudioSettings,
) -> None:
    labeling = extraction_labeling("discont-extract/trigger", _preds, "run-1")
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
    labeling = extraction_labeling("discont-extract/trigger", _preds, "run-1")

    with StudioClient(studio) as client:
        created = client.create("eh30pt-e2e-delete", labeling.config).unwrap()
        assert created.id in {project.id for project in client.projects().unwrap()}

        client.delete(created.id).unwrap()
        assert created.id not in {project.id for project in client.projects().unwrap()}


def test_export_labels_writes_annotations_to_parquet(
    studio: LabelStudioSettings, project: ProjectRef, tmp_path: Path
) -> None:
    import polars as pl

    from eh30pt.cli.scripts.labeling import export_labels
    from eh30pt.labeling.studio import StudioResult

    labeling = extraction_labeling("discont-extract/trigger", _preds, "run-1")
    output = tmp_path / "labels.parquet"

    with StudioClient(studio) as client:
        client.import_tasks(project.id, labeling.tasks).unwrap()
        imported = client.tasks(project.id).unwrap()

        first: StudioResult = {
            "from_name": "discont",
            "to_name": "text",
            "type": "choices",
            "value": {"choices": ["Adverse reactions"]},
        }
        second: StudioResult = {
            "from_name": "meta",
            "to_name": "text",
            "type": "choices",
            "value": {"choices": ["Incomplete phrase"]},
        }
        client.annotate(imported[0]["id"], [first, second]).unwrap()

    assert (
        export_labels(settings=studio, projects=[project.id], output=output) == output
    )

    frame = pl.read_parquet(output)

    assert frame.columns == [
        "task",
        "annotation",
        "text",
        "phrase",
        "drug",
        "reason",
        "discont",
        "other",
        "meta",
    ]
    assert frame.height == 1
    assert frame["discont"].to_list() == [["Adverse reactions"]]
    assert frame["meta"].to_list() == [["Incomplete phrase"]]
    assert frame["drug"].to_list() == ["aspiriini"]


def test_export_labels_combines_projects_into_one_frame(
    studio: LabelStudioSettings, project: ProjectRef, tmp_path: Path
) -> None:
    import polars as pl

    from eh30pt.cli.scripts.labeling import export_labels
    from eh30pt.labeling.studio import StudioResult

    labeling = extraction_labeling("discont-extract/trigger", _preds, "run-1")
    output = tmp_path / "labels.parquet"

    choice: StudioResult = {
        "from_name": "discont",
        "to_name": "text",
        "type": "choices",
        "value": {"choices": ["Treatment success"]},
    }

    with StudioClient(studio) as client:
        other = client.create("eh30pt-e2e-other", labeling.config).unwrap()
        try:
            client.import_tasks(project.id, labeling.tasks[:1]).unwrap()
            client.import_tasks(other.id, labeling.tasks[1:]).unwrap()

            first = client.tasks(project.id).unwrap()
            second = client.tasks(other.id).unwrap()
            client.annotate(first[0]["id"], [choice]).unwrap()
            client.annotate(second[0]["id"], [choice]).unwrap()

            assert (
                export_labels(
                    settings=studio,
                    projects=[project.id, other.id],
                    output=output,
                )
                == output
            )
        finally:
            client.delete(other.id).unwrap()

    frame = pl.read_parquet(output)

    _rows = 2
    assert frame.height == _rows
    assert frame["discont"].to_list() == [["Treatment success"], ["Treatment success"]]
