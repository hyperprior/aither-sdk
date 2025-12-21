# aither

Python SDK for the [Aither](https://aither.computer) platform - contextual intelligence and model observability.

## Features

- **Non-blocking**: Predictions are logged asynchronously without blocking your application
- **Automatic batching**: Multiple predictions are sent in batches for efficiency
- **Zero latency impact**: Background worker handles all API communication
- **Simple API**: Synchronous interface hides async complexity
- **Automatic flushing**: Graceful shutdown ensures no predictions are lost

## Installation

```bash
pip install aither
```

## Quick Start

```python
import aither

# Initialize with your API key
aither.init(api_key="ak_your_api_key")

# Log predictions - returns immediately without blocking!
aither.log_prediction(
    model_id="churn-classifier-v2",
    prediction=0.73,
    features={"tenure": 24, "monthly_charges": 65.5},
    metadata={"user_id": "u_123"}
)

# Predictions are queued and sent in the background
# Your code continues instantly - zero latency impact!
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

### `aither.init(api_key=None, endpoint=None, flush_interval=1.0, batch_size=100)`

Initialize the global client.

- `api_key`: Your Aither API key (or set `AITHER_API_KEY` env var)
- `endpoint`: API endpoint URL (default: `https://aither.computer`)
- `flush_interval`: How often to flush queued predictions in seconds (default: 1.0)
- `batch_size`: Maximum predictions per batch request (default: 100)

### `aither.log_prediction(model_id, prediction, features=None, metadata=None)`

Log a model prediction (non-blocking).

- `model_id`: Identifier for your model (e.g., "churn-classifier-v2")
- `prediction`: The prediction value (float, int, str, or dict)
- `features`: Dictionary of input features (optional)
- `metadata`: Additional context (optional)

**Returns immediately** - prediction is queued and sent in the background.

### `aither.flush()`

Force immediate flush of all queued predictions (blocking).

Useful for:
- Ensuring predictions are sent before shutdown
- Testing
- End of batch processing

```python
aither.log_prediction(model_id="my-model", prediction=0.5)
aither.flush()  # Wait for all predictions to be sent
```

### `aither.close()`

Close the global client and flush remaining predictions.

Called automatically on program exit via `atexit`, but you can call it explicitly:

```python
aither.close()  # Flush and shutdown background worker
```

### `AitherClient`

For more control, use the client class directly:

```python
from aither import AitherClient

# With async background worker (default)
client = AitherClient(
    api_key="ak_your_api_key",
    flush_interval=1.0,
    batch_size=100
)

# Immediate mode (blocking, no background worker)
client = AitherClient(
    api_key="ak_your_api_key",
    enable_background=False  # Send immediately, useful for debugging
)

client.log_prediction(model_id="my-model", prediction=0.5)
client.close()  # Always close when done
```

## FastAPI Integration

```python
import aither
from fastapi import FastAPI

aither.init(api_key="ak_your_api_key")
app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    prediction = model.predict(data)

    # Non-blocking - returns instantly
    aither.log_prediction(
        model_id="my-model",
        prediction=prediction,
        features=data
    )

    return {"prediction": prediction}

@app.on_event("shutdown")
async def shutdown():
    aither.close()  # Flush remaining predictions
```

See `examples/fastapi_example.py` for a complete example.

## License

MIT
