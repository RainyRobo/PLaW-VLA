# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import math
from typing import Generic, TypeVar

T = TypeVar("T")


def resolve_prepare_task_cpus(
    requested: float | None,
    *,
    default: float = 0.5,
) -> float:
    value = float(default if requested is None else requested)
    if value <= 0:
        raise ValueError(f"prepare task CPUs must be > 0, got {value}.")
    return value


def resolve_prepare_concurrency(
    *,
    workers: int,
    prepare_task_cpus: float,
) -> int:
    if workers <= 0:
        return 1
    safe_cpu_per_task = max(float(prepare_task_cpus), 1e-3)
    return max(1, int(math.ceil(float(workers) / safe_cpu_per_task)))


def resolve_max_inflight_episodes(
    requested: int | None,
    *,
    workers: int,
    episode_count: int,
    prepare_task_cpus: float = 1.0,
) -> int:
    if episode_count <= 0:
        return 1
    if requested is None:
        prepare_concurrency = resolve_prepare_concurrency(
            workers=workers,
            prepare_task_cpus=prepare_task_cpus,
        )
        # Keep queue depth above prepare parallelism so prepare can overlap commit I/O.
        requested = max(workers * 2, prepare_concurrency * 2)
    return max(1, min(int(requested), episode_count))


def pop_contiguous_batch(
    buffered_items: dict[int, T],
    *,
    next_index: int,
    batch_size: int,
) -> tuple[list[T], int]:
    batch: list[T] = []
    while len(batch) < batch_size and next_index in buffered_items:
        batch.append(buffered_items.pop(next_index))
        next_index += 1
    return batch, next_index


class OrderedAsyncBatchCommitter(Generic[T]):
    def __init__(
        self,
        commit_fn: Callable[[list[T]], None],
        *,
        on_batch_committed: Callable[[list[T]], None] | None = None,
        max_pending_batches: int = 2,
    ) -> None:
        self._commit_fn = commit_fn
        self._on_batch_committed = on_batch_committed
        self._max_pending_batches = max(1, max_pending_batches)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset-commit")
        self._pending_batches: deque[tuple[Future[None], list[T]]] = deque()

    def __enter__(self) -> OrderedAsyncBatchCommitter[T]:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def submit(self, batch: list[T]) -> int:
        if not batch:
            return 0

        committed = 0
        while len(self._pending_batches) >= self._max_pending_batches:
            committed += self._drain_one(block=True)

        batch_copy = list(batch)
        future = self._executor.submit(self._commit_fn, batch_copy)
        self._pending_batches.append((future, batch_copy))
        committed += self.drain_completed()
        return committed

    def drain_completed(self) -> int:
        committed = 0
        while self._pending_batches and self._pending_batches[0][0].done():
            committed += self._drain_one(block=False)
        return committed

    def close(self) -> int:
        committed = 0
        error: BaseException | None = None
        try:
            while self._pending_batches:
                committed += self._drain_one(block=True)
        except BaseException as exc:
            error = exc
        finally:
            self._executor.shutdown(wait=True, cancel_futures=error is not None)
        if error is not None:
            raise error
        return committed

    def _drain_one(self, *, block: bool) -> int:
        if not self._pending_batches:
            return 0

        future, batch = self._pending_batches[0]
        if not block and not future.done():
            return 0

        future.result()
        self._pending_batches.popleft()
        if self._on_batch_committed is not None:
            self._on_batch_committed(batch)
        return len(batch)
