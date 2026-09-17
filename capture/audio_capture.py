"""
System Audio Capture — Captures system audio output via WASAPI loopback.
This captures what the OTHER person on the call is saying (not your mic).
"""

import numpy as np
import sounddevice as sd
import threading
import queue
import time

from config import SAMPLE_RATE, AUDIO_WINDOW_SECONDS, AUDIO_STRIDE_SECONDS


class SystemAudioCapture:
    """
    Captures system audio output using WASAPI loopback on Windows.
    This records what you HEAR (speakers/headphones), which is the caller's voice.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.window_size = int(AUDIO_WINDOW_SECONDS * sample_rate)
        self.stride_size = int(AUDIO_STRIDE_SECONDS * sample_rate)

        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.is_running = False
        self._audio_queue = queue.Queue()
        self._stream = None
        self._thread = None
        self._device_id = None

    def find_loopback_device(self) -> int:
        """
        Find a WASAPI loopback device for system audio capture.
        Returns the device ID or -1 if not found.
        """
        devices = sd.query_devices()
        # Look for loopback devices (Windows WASAPI)
        for i, dev in enumerate(devices):
            name = dev["name"].lower()
            if "loopback" in name or "stereo mix" in name or "what u hear" in name:
                print(f"[AudioCapture] Found loopback device: {dev['name']} (id={i})")
                return i

        # Try to find the default output device and use it as loopback
        try:
            hostapis = sd.query_hostapis()
            for api in hostapis:
                if "wasapi" in api["name"].lower():
                    default_output = api.get("default_output_device", -1)
                    if default_output >= 0:
                        print(f"[AudioCapture] Using WASAPI default output as loopback (id={default_output})")
                        return default_output
        except Exception:
            pass

        print("[AudioCapture] No loopback device found — will use microphone fallback")
        return -1

    def list_devices(self) -> list:
        """List all available audio devices."""
        devices = sd.query_devices()
        device_list = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                device_list.append({
                    "id": i,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "sample_rate": dev["default_samplerate"],
                })
        return device_list

    def set_device(self, device_id: int):
        """Manually set the audio capture device."""
        self._device_id = device_id
        dev = sd.query_devices(device_id)
        print(f"[AudioCapture] Device set: {dev['name']}")

    def start(self, use_loopback: bool = True):
        """
        Start capturing system audio.

        Args:
            use_loopback: If True, try to capture system audio output.
                         If False, use default microphone.
        """
        if self.is_running:
            return

        self.is_running = True
        self.audio_buffer = np.zeros(0, dtype=np.float32)

        # Find capture device
        if self._device_id is None:
            if use_loopback:
                self._device_id = self.find_loopback_device()

        device = self._device_id if self._device_id >= 0 else None

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                device=device,
                callback=self._audio_callback,
            )
            self._stream.start()

            # Start buffer processing thread
            self._thread = threading.Thread(target=self._buffer_loop, daemon=True)
            self._thread.start()

            dev_name = sd.query_devices(device)["name"] if device else "default"
            print(f"[AudioCapture] Started on: {dev_name}")

        except Exception as e:
            print(f"[AudioCapture] Failed to start: {e}")
            print("[AudioCapture] Falling back to default microphone")
            self._device_id = None
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=int(self.sample_rate * 0.1),
                callback=self._audio_callback,
            )
            self._stream.start()
            self._thread = threading.Thread(target=self._buffer_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop audio capture."""
        self.is_running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[AudioCapture] Stopped")

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            pass  # Suppress status messages during normal operation
        self._audio_queue.put(indata[:, 0].copy())

    def _buffer_loop(self):
        """Collect audio chunks into a rolling buffer."""
        while self.is_running:
            try:
                while not self._audio_queue.empty():
                    chunk = self._audio_queue.get_nowait()
                    self.audio_buffer = np.concatenate([self.audio_buffer, chunk])

                # Keep buffer from growing too large (keep last 10 seconds)
                max_buffer = self.sample_rate * 10
                if len(self.audio_buffer) > max_buffer:
                    self.audio_buffer = self.audio_buffer[-max_buffer:]

                time.sleep(0.05)
            except Exception:
                time.sleep(0.1)

    def get_latest_window(self) -> np.ndarray:
        """
        Get the latest audio window for analysis.
        Returns None if not enough audio buffered yet.
        """
        if len(self.audio_buffer) >= self.window_size:
            return self.audio_buffer[-self.window_size:].copy()
        return None

    def get_audio_level(self) -> float:
        """Get current audio level (RMS) for visualization."""
        if len(self.audio_buffer) > 0:
            recent = self.audio_buffer[-int(self.sample_rate * 0.1):]
            return float(np.sqrt(np.mean(recent ** 2)))
        return 0.0
