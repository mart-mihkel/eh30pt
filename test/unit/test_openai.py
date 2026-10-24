"""OpenAI wrangling tests."""

import asyncio

import pytest

from eh30pt.openai.stream import (
    append_prediction_window,
    batched_concurrent,
    map_concurrent,
    maybe_log_invalid,
)
from eh30pt.types import FewShot, GenerateRecord, Predictor


class _Identity(Predictor[int, int]):
    async def predict(self, item: int) -> int:
        return item


class _SlowFirst(Predictor[int, int]):
    async def predict(self, item: int) -> int:
        await asyncio.sleep(0.05 if item == 1 else 0)
        return item


class _SlowOdd(Predictor[int, int]):
    async def predict(self, item: int) -> int:
        await asyncio.sleep(0.02 if item % 2 else 0)
        return item * 2


def test_map_concurrent_empty() -> None:
    assert asyncio.run(map_concurrent([], _Identity(), concurrency=4)) == []


def test_map_concurrent_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        asyncio.run(map_concurrent([1], _Identity(), concurrency=0))


def test_map_concurrent_preserves_input_order() -> None:
    async def _main() -> list[int]:
        return await map_concurrent([1, 2, 3], _SlowFirst(), concurrency=3)

    assert asyncio.run(_main()) == [1, 2, 3]


def test_map_concurrent_bounds_in_flight() -> None:
    state = {"active": 0, "peak": 0}
    expected_peak = 2

    class _Tracked(Predictor[int, int]):
        async def predict(self, _: int) -> int:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.02)
            state["active"] -= 1
            return 0

    async def _main(concurrency: int) -> None:
        await map_concurrent([0, 0, 0, 0], _Tracked(), concurrency=concurrency)

    asyncio.run(_main(expected_peak))
    assert state["peak"] == expected_peak

    state["peak"] = 0
    asyncio.run(_main(1))
    assert state["peak"] == 1


def test_batched_concurrent_rejects_non_positive() -> None:
    async def _collect() -> None:
        async for _ in batched_concurrent(iter([1]), _Identity(), concurrency=0):
            pass

    with pytest.raises(ValueError, match="concurrency"):
        asyncio.run(_collect())


def test_batched_concurrent_windows_stream_in_order() -> None:
    async def _main() -> list[tuple[list[int], list[int]]]:
        return [
            batch
            async for batch in batched_concurrent(
                iter(range(5)), _SlowOdd(), concurrency=2
            )
        ]

    assert asyncio.run(_main()) == [
        ([0, 1], [0, 2]),
        ([2, 3], [4, 6]),
        ([4], [8]),
    ]


def test_log_invalid_silent_when_clean(
    caplog: pytest.LogCaptureFixture,
) -> None:
    maybe_log_invalid(0, 10)

    assert "invalid responses" not in caplog.text


def test_log_invalid_warns_with_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    maybe_log_invalid(2, 10)

    assert "2/10 invalid responses" in caplog.text


def test_append_prediction_window() -> None:
    batch: list[GenerateRecord] = []
    window = [
        FewShot(user="a", label="x", id=0),
        FewShot(user="b", label="y", id=1),
    ]

    invalid = append_prediction_window(batch, window, ["x", None])

    assert invalid == 1
    assert batch == [
        GenerateRecord(id=0, user="a", prediction="x"),
        GenerateRecord(id=1, user="b", prediction="<invalid>"),
    ]


def test_append_prediction_window_uses_sample_id() -> None:
    batch: list[GenerateRecord] = []
    window = [
        FewShot(user="a", label="", id=42),
        FewShot(user="b", label="", id=107),
    ]

    invalid = append_prediction_window(batch, window, ["x", "y"])

    assert invalid == 0
    assert batch == [
        GenerateRecord(id=42, user="a", prediction="x"),
        GenerateRecord(id=107, user="b", prediction="y"),
    ]


def test_append_prediction_window_rejects_missing_id() -> None:
    batch: list[GenerateRecord] = []
    window = [FewShot(user="a", label="x")]

    with pytest.raises(ValueError, match="row ids"):
        append_prediction_window(batch, window, ["x"])


def test_asafe_wraps_success_and_failure() -> None:
    import asyncio

    from returns.result import Failure, Success

    from eh30pt.types import asafe

    @asafe
    async def _ok(value: int) -> int:
        return value

    @asafe
    async def _boom() -> int:
        msg = "boom"
        raise RuntimeError(msg)

    assert asyncio.run(_ok(3)) == Success(3)
    assert isinstance(asyncio.run(_boom()), Failure)
