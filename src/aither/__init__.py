"""Aither SDK - Python client for the Aither platform."""

from __future__ import annotations

from typing import Any

from aither.client import AitherClient

__version__ = "0.1.0"
__all__ = ["AitherClient", "init", "log_prediction"]

_client: AitherClient | None = None


def init(
    api_key: str | None = None,
    endpoint: str | None = None,
) -> None:
    """Initialize the global Aither client.

    Args:
        api_key: API key for authentication. Falls back to AITHER_API_KEY env var.
        endpoint: API endpoint URL. Falls back to AITHER_ENDPOINT env var or default.
    """
    global _client
    _client = AitherClient(api_key=api_key, endpoint=endpoint)


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
) -> dict[str, Any]:
    """Log a model prediction using the global client.

    Args:
        model_id: Identifier for the model.
        prediction: The prediction value.
        features: Input features used for the prediction.
        metadata: Additional context or metadata.

    Returns:
        Response from the API.
    """
    return _get_client().log_prediction(
        model_id=model_id,
        prediction=prediction,
        features=features,
        metadata=metadata,
    )
