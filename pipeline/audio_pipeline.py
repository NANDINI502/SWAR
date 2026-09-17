"""
Audio Pipeline — Real-time audio capture and deepfake voice detection.
Processes audio in sliding windows for continuous monitoring.
"""

import numpy as np
import sounddevice as sd
import threading
import queue
import time
import librosa

from config import SAMPLE_RATE, AUDIO_WINDOW_SECONDS, AUDIO_STRIDE_SECONDS, DEVICE
from models.audio_detector import AudioDetectorInference


class AudioPipeline:
    """
    Real-time audio deepfake detection pipeline.
    Captures microphone input, buffers into windows, and classifies.
    """

    def __init__(self, device: str = DEVICE, use_onnx: bool = False):
        self.detector = AudioDetectorInference(device=device)
        self.sample_rate = SAMPLE_RATE
        self.window_size = int(AUDIO_WINDOW_SECONDS * SAMPLE_RATE)
        self.stride_size = int(AUDIO_STRIDE_SECONDS * SAMPLE_RATE)

        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.last_result = {"label": "WAITING...", "confidence": 0.0, "fake_probability": 0.0}
        self.is_running = False
        self._audio_queue = queue.Queue()
        self._stream = None
        self._thread = None

        # Optional third signal (roadmap item 1) — set externally, e.g.
        # `audio_pipeline.lipsync = LipSyncConsistency()`. When unset,
        # the mic loop behaves exactly as before.
        self.lipsync = None

        # Optional DSP forensic-cue fusion (roadmap item 3) — set
        # externally, e.g. `audio_pipeline.audio_dsp =
        # AudioDSPFeatureExtractor()`. When unset, predictions are
        # Wav2Vec2's own output, unchanged from before this item.
        self.audio_dsp = None

    def start_stream(self):
        """Start capturing audio from the microphone."""
        if self.is_running:
            return

        self.is_running = True
        self.audio_buffer = np.zeros(0, dtype=np.float32)

        # Start audio input stream
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
            callback=self._audio_callback,
        )
        self._stream.start()

        # Start processing thread
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

        print("[AudioPipeline] Microphone stream started")

    def stop_stream(self):
        """Stop audio capture."""
        self.is_running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[AudioPipeline] Microphone stream stopped")

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice — pushes audio chunks to queue."""
        if status:
            print(f"[AudioPipeline] Stream status: {status}")
        self._audio_queue.put(indata[:, 0].copy())

    def _process_loop(self):
        """Background thread: collect audio chunks and run detection."""
        while self.is_running:
            try:
                # Collect audio from queue
                while not self._audio_queue.empty():
                    chunk = self._audio_queue.get_nowait()
                    self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
                    if self.lipsync is not None:
                        self.lipsync.update_audio(chunk, self.sample_rate)

                # Process when we have enough audio
                if len(self.audio_buffer) >= self.window_size:
                    window = self.audio_buffer[:self.window_size]
                    result = self.detector.predict(window)
                    if self.audio_dsp is not None:
                        result = self.audio_dsp.fuse(result, window, self.sample_rate)
                    self.last_result = result

                    # Slide buffer forward
                    self.audio_buffer = self.audio_buffer[self.stride_size:]

                time.sleep(0.05)  # Small delay to prevent busy-waiting

            except Exception as e:
                print(f"[AudioPipeline] Error: {e}")
                time.sleep(0.1)

    def get_current_result(self) -> dict:
        """Get the latest detection result."""
        return self.last_result

    def process_audio_file(self, file_path: str) -> list[dict]:
        """
        Process an uploaded audio file.

        Args:
            file_path: Path to audio file (WAV, MP3, etc.)

        Returns:
            List of detection results for each window
        """
        # Load audio file
        waveform, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

        results = []
        offset = 0

        while offset + self.window_size <= len(waveform):
            window = waveform[offset:offset + self.window_size]
            result = self.detector.predict(window)
            if self.audio_dsp is not None:
                result = self.audio_dsp.fuse(result, window, self.sample_rate)
            result["time_start"] = round(offset / self.sample_rate, 2)
            result["time_end"] = round((offset + self.window_size) / self.sample_rate, 2)
            results.append(result)
            offset += self.stride_size

        # Process remaining audio if any
        if offset < len(waveform) and len(waveform) - offset > self.sample_rate:
            remaining = waveform[offset:]
            # Pad to window size
            padded = np.pad(remaining, (0, self.window_size - len(remaining)))
            result = self.detector.predict(padded)
            if self.audio_dsp is not None:
                result = self.audio_dsp.fuse(result, padded, self.sample_rate)
            result["time_start"] = round(offset / self.sample_rate, 2)
            result["time_end"] = round(len(waveform) / self.sample_rate, 2)
            results.append(result)

        return results

    def process_audio_file_from_array(self, waveform: np.ndarray) -> list:
        """
        Process audio from a numpy array (e.g., from Gradio's microphone).

        Args:
            waveform: 1D numpy array of audio samples at self.sample_rate

        Returns:
            List of detection results for each window
        """
        results = []
        offset = 0

        while offset + self.window_size <= len(waveform):
            window = waveform[offset:offset + self.window_size]
            result = self.detector.predict(window)
            if self.audio_dsp is not None:
                result = self.audio_dsp.fuse(result, window, self.sample_rate)
            result["time_start"] = round(offset / self.sample_rate, 2)
            result["time_end"] = round((offset + self.window_size) / self.sample_rate, 2)
            results.append(result)
            offset += self.stride_size

        return results

    def get_audio_level(self) -> float:
        """Get current audio input level (RMS) for visualization."""
        if len(self.audio_buffer) > 0:
            recent = self.audio_buffer[-int(self.sample_rate * 0.1):]  # Last 100ms
            return float(np.sqrt(np.mean(recent ** 2)))
        return 0.0
