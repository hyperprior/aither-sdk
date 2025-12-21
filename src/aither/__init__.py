"""Aither SDK - Python client for the Aither platform."""

from __future__ import annotations

from typing import Any

from aither.client import AitherClient

__version__ = "0.1.0"
__all__ = ["AitherClient", "init", "log_prediction", "flush", "close"]

_client: AitherClient | None = None


def init(
    api_key: str | None = None,
    endpoint: str | None = None,
    flush_interval: float = 1.0,
    batch_size: int = 100,
) -> None:
    """Initialize the global Aither client.

    Args:
        api_key: API key for authentication. Falls back to AITHER_API_KEY env var.
        endpoint: API endpoint URL. Falls back to AITHER_ENDPOINT env var or default.
        flush_interval: How often to flush queued predictions (seconds).
        batch_size: Maximum predictions per batch request.
    """
    global _client
    _client = AitherClient(
        api_key=api_key,
        endpoint=endpoint,
        flush_interval=flush_interval,
        batch_size=batch_size,
    )


def _get_client() -> AitherClient:
    """Get or create the global client."""
    global _client
    if _client is None:
        _client = AitherClient()
    return _client


def log_prediction(
    model_id: str,
    prediction: Any,
    features: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log a model prediction using the global client (non-blocking).

    Predictions are queued and sent asynchronously in the background.
    This function returns immediately without waiting for the API response.

    Args:
        model_id: Identifier for the model.
        prediction: The prediction value.
        features: Input features used for the prediction.
        metadata: Additional context or metadata.
    """
    _get_client().log_prediction(
        model_id=model_id,
        prediction=prediction,
        features=features,
        metadata=metadata,
    )


def flush() -> None:
    """Force immediate flush of queued predictions (blocking).

    Useful for ensuring predictions are sent before shutdown or in tests.
    """
    _get_client().flush()


def close() -> None:
    """Close the global client and flush remaining predictions."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
