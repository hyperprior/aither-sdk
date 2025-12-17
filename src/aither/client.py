"""Aither client implementation."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_ENDPOINT = "https://aither.computer"


class AitherClient:
    """Client for the Aither platform API."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the Aither client.

        Args:
            api_key: API key for authentication. Falls back to AITHER_API_KEY env var.
            endpoint: API endpoint URL. Falls back to AITHER_ENDPOINT env var or default.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("AITHER_API_KEY")
        self.endpoint = (
            endpoint or os.environ.get("AITHER_ENDPOINT") or DEFAULT_ENDPOINT
        )
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.endpoint,
            timeout=timeout,
            headers=self._build_headers(),
        )

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def log_prediction(
        self,
        model_id: str,
        prediction: Any,
        features: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Log a model prediction.

        Args:
            model_id: Identifier for the model.
            prediction: The prediction value.
            features: Input features used for the prediction.
            metadata: Additional context or metadata.

        Returns:
            Response from the API.
        """
        payload = {
            "model_id": model_id,
            "prediction": prediction,
        }
        if features is not None:
            payload["features"] = features
        if metadata is not None:
            payload["metadata"] = metadata

        response = self._client.post("/v1/predictions", json=payload)
        response.raise_for_status()
        return response.json()

    def health(self) -> bool:
        """Check if the API is healthy.

        Returns:
            True if the API is healthy.
        """
        response = self._client.get("/health")
        return response.status_code == 200

    def close(self) -> None:
        """Close the client connection."""
        self._client.close()

    def __enter__(self) -> AitherClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
