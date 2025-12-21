"""Aither client implementation."""

from __future__ import annotations

import asyncio
import atexit
import os
import threading
from collections import deque
from typing import Any

import httpx

DEFAULT_ENDPOINT = "https://aither.computer"
DEFAULT_FLUSH_INTERVAL = 1.0  # seconds
DEFAULT_BATCH_SIZE = 100


class AitherClient:
    """Client for the Aither platform API.

    Uses async httpx internally with a background worker for non-blocking operation.
    The API is synchronous for ease of use - predictions are queued and flushed
    asynchronously in the background.
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        enable_background: bool = True,
    ) -> None:
        """Initialize the Aither client.

        Args:
            api_key: API key for authentication. Falls back to AITHER_API_KEY env var.
            endpoint: API endpoint URL. Falls back to AITHER_ENDPOINT env var or default.
            timeout: Request timeout in seconds.
            flush_interval: How often to flush queued predictions (seconds).
            batch_size: Maximum predictions per batch request.
            enable_background: If False, predictions are sent immediately (blocking).
        """
        self.api_key = api_key or os.environ.get("AITHER_API_KEY")
        self.endpoint = (
            endpoint or os.environ.get("AITHER_ENDPOINT") or DEFAULT_ENDPOINT
        )
        self.timeout = timeout
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.enable_background = enable_background

        # Thread-safe queue for predictions
        self._queue: deque[dict[str, Any]] = deque()
        self._queue_lock = threading.Lock()

        # Background worker management
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

        # Start background worker
        if self.enable_background:
            self._start_worker()
            atexit.register(self.close)

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _start_worker(self) -> None:
        """Start the background worker thread."""
        self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self._worker_thread.start()

    def _run_worker(self) -> None:
        """Worker thread that runs async event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._worker_loop())
        finally:
            self._loop.close()

    async def _worker_loop(self) -> None:
        """Async worker loop that periodically flushes the queue."""
        async with httpx.AsyncClient(
            base_url=self.endpoint,
            timeout=self.timeout,
            headers=self._build_headers(),
        ) as client:
            while not self._stop_event.is_set():
                try:
                    await self._flush_queue(client)
                except Exception as e:
                    # Log errors but keep the worker running
                    # TODO: Add proper logging
                    print(f"Error flushing queue: {e}")

                # Wait for flush interval or stop event
                try:
                    await asyncio.wait_for(
                        asyncio.shield(asyncio.create_task(asyncio.sleep(float('inf')))),
                        timeout=self.flush_interval
                    )
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue to next flush

    async def _flush_queue(self, client: httpx.AsyncClient) -> None:
        """Flush predictions from queue to API."""
        predictions_to_send = []

        with self._queue_lock:
            # Take up to batch_size predictions from queue
            while self._queue and len(predictions_to_send) < self.batch_size:
                predictions_to_send.append(self._queue.popleft())

        if not predictions_to_send:
            return

        # Send batch if we have multiple predictions, otherwise send single
        if len(predictions_to_send) == 1:
            await client.post("/v1/predictions", json=predictions_to_send[0])
        else:
            # Use batch endpoint if available
            await client.post("/v1/predictions/batch", json={"predictions": predictions_to_send})

    def log_prediction(
        self,
        model_id: str,
        prediction: Any,
        features: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a model prediction (non-blocking).

        Args:
            model_id: Identifier for the model.
            prediction: The prediction value.
            features: Input features used for the prediction.
            metadata: Additional context or metadata.
        """
        payload = {
            "model_id": model_id,
            "prediction": prediction,
        }
        if features is not None:
            payload["features"] = features
        if metadata is not None:
            payload["metadata"] = metadata

        if self.enable_background:
            # Add to queue for async processing
            with self._queue_lock:
                self._queue.append(payload)
        else:
            # Immediate mode: block and send synchronously
            with httpx.Client(
                base_url=self.endpoint,
                timeout=self.timeout,
                headers=self._build_headers(),
            ) as client:
                response = client.post("/v1/predictions", json=payload)
                response.raise_for_status()

    def flush(self) -> None:
        """Force immediate flush of queued predictions (blocking).

        Useful for ensuring predictions are sent before shutdown or in tests.
        """
        if not self.enable_background or not self._loop:
            return

        # Run flush in the worker's event loop
        future = asyncio.run_coroutine_threadsafe(
            self._flush_queue_sync(),
            self._loop
        )
        future.result(timeout=self.timeout)

    async def _flush_queue_sync(self) -> None:
        """Helper to flush queue from external thread."""
        async with httpx.AsyncClient(
            base_url=self.endpoint,
            timeout=self.timeout,
            headers=self._build_headers(),
        ) as client:
            await self._flush_queue(client)

    def health(self) -> bool:
        """Check if the API is healthy.

        Returns:
            True if the API is healthy.
        """
        with httpx.Client(base_url=self.endpoint, timeout=self.timeout) as client:
            response = client.get("/health")
            return response.status_code == 200

    def close(self) -> None:
        """Close the client and flush remaining predictions."""
        if not self.enable_background:
            return

        # Signal worker to stop
        self._stop_event.set()

        # Flush any remaining predictions
        try:
            self.flush()
        except Exception:
            pass  # Best effort

        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

    def __enter__(self) -> AitherClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
