# aither

Python SDK for the [Aither](https://aither.computer) platform - contextual intelligence and model observability.

## Installation

```bash
pip install aither
```

## Quick Start

```python
import aither

# Initialize with your API key
aither.init(api_key="ak_your_api_key")

# Log a prediction
aither.log_prediction(
    model_id="churn-classifier-v2",
    prediction=0.73,
    features={"tenure": 24, "monthly_charges": 65.5},
    metadata={"user_id": "u_123"}
)
```

## Configuration

### Environment Variables

```bash
export AITHER_API_KEY="ak_your_api_key"
export AITHER_ENDPOINT="https://aither.computer"  # optional
```

### Explicit Initialization

```python
import aither

aither.init(
    api_key="ak_your_api_key",
    endpoint="https://aither.computer"
)
```

## API Reference

### `aither.init(api_key=None, endpoint=None)`

Initialize the global client.

- `api_key`: Your Aither API key (or set `AITHER_API_KEY` env var)
- `endpoint`: API endpoint URL (default: `https://aither.computer`)

### `aither.log_prediction(model_id, prediction, features=None, metadata=None)`

Log a model prediction.

- `model_id`: Identifier for your model (e.g., "churn-classifier-v2")
- `prediction`: The prediction value (float, int, str, or dict)
- `features`: Dictionary of input features (optional)
- `metadata`: Additional context (optional)

### `AitherClient`

For more control, use the client class directly:

```python
from aither import AitherClient

client = AitherClient(api_key="ak_your_api_key")
client.log_prediction(
    model_id="my-model",
    prediction=0.5
)
```

## License

MIT
