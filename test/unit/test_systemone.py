"""System One decision request tests."""

import asyncio

from eh30pt.settings import ServerSettings
from eh30pt.systemone.client import SystemOneClient


def test_payload_names_model_and_labels() -> None:
    payload = SystemOneClient.build_payload(
        state="I was charged twice.",
        instructions="Choose the category.",
        labels=("billing", "shipping"),
        model="kev",
    )

    assert payload.model == "kev"
    assert payload.questions["label"].criteria == {
        "billing": None,
        "shipping": None,
    }


def test_client_uses_configured_model_without_listing() -> None:
    client = SystemOneClient(ServerSettings(model="kev"))

    assert asyncio.run(client.model()).unwrap() == "kev"
