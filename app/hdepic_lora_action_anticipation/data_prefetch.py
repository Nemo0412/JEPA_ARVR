"""Async dataloader prefetch to keep the GPU busy while video workers decode.

Cluster util-kill (~2h, AveUtil < 60%) is usually caused by a synchronous
``next(loader)`` on the main thread: H100 sits idle for seconds between steps.
A background producer fills a small queue so ``get()`` is usually ready.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class DataLoaderPrefetcher:
    """Background ``next(data_loader)`` with a bounded in-flight queue."""

    def __init__(self, data_loader, *, depth: int | None = None, name: str = "dl-prefetch"):
        self.data_loader = data_loader
        if depth is None:
            depth = int(os.environ.get("DATALOADER_PREFETCH_DEPTH", "4") or "4")
        self.depth = max(1, int(depth))
        self._q: queue.Queue = queue.Queue(maxsize=self.depth)
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        logger.info("DataLoaderPrefetcher started: depth=%d name=%s", self.depth, name)

    def _run(self) -> None:
        _it = iter(self.data_loader)
        while not self._stop.is_set():
            t0 = time.time()
            try:
                try:
                    udata = next(_it)
                except Exception:
                    _it = iter(self.data_loader)
                    udata = next(_it)
                fetch_ms = (time.time() - t0) * 1000.0
                self._q.put((udata, fetch_ms))
            except BaseException as exc:  # noqa: BLE001 — surface to consumer
                self._error = exc
                self._q.put(None)
                return

    def get(self) -> tuple[Any, float]:
        item = self._q.get()
        if item is None:
            if self._error is not None:
                raise RuntimeError("DataLoaderPrefetcher failed") from self._error
            raise RuntimeError("DataLoaderPrefetcher stopped unexpectedly")
        return item

    def close(self) -> None:
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5.0)
