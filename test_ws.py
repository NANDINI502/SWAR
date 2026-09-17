import asyncio
import websockets
import numpy as np

async def test_ws():
    uri = "ws://127.0.0.1:8000/ws/audio"
    async with websockets.connect(uri) as ws:
        # Send 3 seconds of zeros (should be FAKE)
        audio_data = np.zeros(16000 * 3, dtype=np.float32)
        await ws.send(audio_data.tobytes())
        response = await ws.recv()
        print(f"Zeros response: {response}")

        # Send 3 seconds of sine wave (should be FAKE)
        t = np.linspace(0, 3, 16000 * 3)
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        await ws.send(sine.tobytes())
        response = await ws.recv()
        print(f"Sine response: {response}")

asyncio.run(test_ws())
