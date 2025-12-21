"""Quickstart example - minimal code to get started with Aither SDK."""

import aither

# Set your API key (or use AITHER_API_KEY environment variable)
aither.init(api_key="ak_your_api_key_here")

# Log a simple prediction - that's it!
aither.log_prediction(
    model_id="my-first-model",
    prediction="success",
)

print("Prediction logged! It will be sent to Aither in the background.")
print("The function returned immediately without waiting for the HTTP request.")

# Optional: wait for it to be sent
aither.flush()
print("Flushed - prediction has been sent!")
