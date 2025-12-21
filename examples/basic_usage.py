"""Basic usage example for Aither SDK.

This demonstrates the simplest way to use the SDK - predictions are logged
asynchronously in the background without blocking your code.
"""

import time
import aither

# Initialize the client (optional - will auto-initialize with env vars if not called)
aither.init(api_key="ak_your_api_key_here")

# Log predictions - these return immediately without blocking
print("Logging predictions (non-blocking)...")

for i in range(10):
    aither.log_prediction(
        model_id="sentiment-classifier-v1",
        prediction="positive" if i % 2 == 0 else "negative",
        features={
            "text_length": 100 + i * 10,
            "has_emoji": bool(i % 3),
        },
        metadata={
            "request_id": f"req_{i}",
            "user_id": f"user_{i % 3}",
        },
    )
    print(f"  Logged prediction {i+1}/10 - instantly returned!")

print("\nAll predictions logged and queued for sending.")
print("They'll be sent in batches every ~1 second in the background.")
print("\nYour code can continue immediately without waiting...")

# Simulate other work
time.sleep(0.5)
print("Doing other work while predictions are being sent...")

# Optional: Force flush before exiting (happens automatically on exit anyway)
print("\nFlushing remaining predictions...")
aither.flush()

print("Done! All predictions sent.")
