"""Tests for reading annotations back from the Label Studio API."""

import pytest

from eh30pt.labeling.studio import (
    StudioClient,
    _number,
    _object,
    _results,
    _tasks,
    annotation_frame,
)
from eh30pt.settings import LabelStudioSettings


def test_authorization_sends_a_legacy_token_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LABEL_STUDIO_TOKEN", "dev-token")

    with StudioClient(LabelStudioSettings()) as client:
        assert client._authorization() == "Token dev-token"


def test_authorization_sends_a_cached_access_token_as_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LABEL_STUDIO_TOKEN", "dev-token")
    monkeypatch.setenv("LABEL_STUDIO_AUTH", "refresh")

    with StudioClient(LabelStudioSettings()) as client:
        client._access = "short-lived"

        assert client._authorization() == "Bearer short-lived"


def test_object_rejects_a_non_object_payload() -> None:
    with pytest.raises(TypeError, match="not a json object"):
        _object([1, 2], "the payload")


def test_number_rejects_a_missing_or_boolean_field() -> None:
    with pytest.raises(TypeError, match="no integer 'id'"):
        _number({}, "id", "a project")

    with pytest.raises(TypeError, match="no integer 'id'"):
        _number({"id": True}, "id", "a project")


def test_results_and_tasks_read_their_own_page_keys() -> None:
    assert _results({"results": [{"id": 1}]}, "a page") == [{"id": 1}]
    assert _tasks({"tasks": [{"id": 2}]}, "a page") == [{"id": 2}]

    with pytest.raises(TypeError, match="no 'results' list"):
        _results({"tasks": []}, "a page")

    with pytest.raises(TypeError, match="no 'tasks' list"):
        _tasks({"results": []}, "a page")


def test_annotation_frame_groups_choices_by_control() -> None:
    tasks = [
        {
            "id": 7,
            "inner_id": 1,
            "data": {"text": "t", "phrase": "p", "drug": "d", "reason": "r"},
            "annotations": [
                {
                    "id": 3,
                    "result": [
                        {
                            "from_name": "discont",
                            "to_name": "text",
                            "type": "choices",
                            "value": {"choices": ["Other"]},
                        },
                        {
                            "from_name": "span",
                            "to_name": "text",
                            "type": "labels",
                            "value": {
                                "start": 0,
                                "end": 1,
                                "text": "t",
                                "labels": ["drug"],
                            },
                        },
                    ],
                },
                {
                    "id": 4,
                    "result": [
                        {
                            "from_name": "unknown",
                            "to_name": "text",
                            "type": "choices",
                            "value": {"choices": ["Whatever"]},
                        },
                    ],
                },
            ],
        },
    ]

    frame = annotation_frame(tasks, ("discont", "other", "meta"))  # ty: ignore[invalid-argument-type]

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
    assert frame["task"].to_list() == [7, 7]
    assert frame["annotation"].to_list() == [3, 4]
    assert frame["discont"].to_list() == [["Other"], []]
    assert frame["other"].to_list() == [[], []]
