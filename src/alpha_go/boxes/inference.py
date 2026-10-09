"""Cross-game batched inference for plane inputs.

Many game threads submit `(K, H, W)` plane stacks; one worker thread stacks them, runs the
net once, and resolves each future with that board's edge logits, win probability and
expected margin. Mirrors `alpha_go.inference.LocalBatchedInferenceEngine` for Go, without
the Go-specific padding.
"""
# ruff: noqa: N803, N806, N815  (dimension-suffixed tensor names)
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from alpha_go.boxes.model import BoxesNet

Result = tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
# per chunk: edge logits (N, E), win prob (N,), expected margin (N,)


@dataclass
class _Request:
    planes_NKHW: NDArray[np.float32]  # one chunk (a tree's leaf batch)
    future: Future[Result]


class PlaneBatchedEngine:
    """Batches plane chunks across threads into single forward passes.

    A chunk is one tree's leaf batch (N, K, H, W); the engine concatenates chunks from
    many game threads up to `batch_size` positions per forward and answers each chunk
    with one future, so the per-leaf Python work is a few array slices.
    """

    def __init__(
        self,
        model: BoxesNet,
        device: torch.device,
        batch_size: int = 64,
        batch_timeout_ms: float = 1.0,
        use_fp16: bool | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0
        self.use_fp16 = device.type == "cuda" if use_fp16 is None else use_fp16
        self.queue: queue.Queue[_Request] = queue.Queue()
        self.running = False
        self.thread: threading.Thread | None = None
        self.total_requests = 0
        self.total_batches = 0

    def footprint_bytes(self) -> int:
        """Bytes of one full input batch (what the GPU holds per forward, before activations)."""
        H, W = 2 * self.model.rows + 1, 2 * self.model.cols + 1
        return self.batch_size * 11 * H * W * 4

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(
            f"PlaneBatchedEngine: device={self.device} batch_size={self.batch_size} "
            f"fp16={self.use_fp16} input batch={self.footprint_bytes() / 1e6:.2f} MB",
            flush=True,
        )

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=5.0)

    def submit(self, planes_NKHW: NDArray[np.float32]) -> Future[Result]:
        """Queue one chunk; the future resolves to (logits_NE, win_N, expected_margin_N)."""
        future: Future[Result] = Future()
        self.queue.put(_Request(planes_NKHW, future))
        return future

    def _loop(self) -> None:
        while self.running:
            batch: list[_Request] = []
            first = self._get(1.0)
            if first is None:
                continue
            batch.append(first)
            deadline = time.perf_counter() + self.batch_timeout
            size = len(first.planes_NKHW)
            while size < self.batch_size:
                request = self._get(max(0.0, deadline - time.perf_counter()))
                if request is None:
                    break
                batch.append(request)
                size += len(request.planes_NKHW)
            self._process(batch)
            self.total_requests += size
            self.total_batches += 1

    def _get(self, timeout: float) -> _Request | None:
        if timeout <= 0 and self.queue.empty():
            return None
        return _queue_get(self.queue, timeout)

    @torch.no_grad()
    def _process(self, batch: list[_Request]) -> None:
        planes_BKHW = torch.from_numpy(np.concatenate([r.planes_NKHW for r in batch]))
        planes_BKHW = planes_BKHW.to(self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_fp16):
            policy_BE, margin_BM = self.model(planes_BKHW)
        policy = policy_BE.float().cpu().numpy()
        win = self.model.win_prob(margin_BM).cpu().numpy()
        expected = self.model.expected_margin(margin_BM).cpu().numpy()
        start = 0
        for request in batch:
            end = start + len(request.planes_NKHW)
            request.future.set_result((policy[start:end], win[start:end], expected[start:end]))
            start = end


def _queue_get(q: queue.Queue[_Request], timeout: float) -> _Request | None:
    """Blocking get that returns None on timeout instead of raising."""
    deadline = time.perf_counter() + timeout
    while True:
        if not q.empty():
            return q.get()
        if time.perf_counter() >= deadline:
            return None
        time.sleep(0.0002)
