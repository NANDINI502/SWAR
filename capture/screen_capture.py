"""
Screen Capture Module — Captures a user-selected screen region for video call monitoring.
Uses mss for fast, low-overhead screen capture.
"""

import numpy as np
import cv2
import threading
import time
from mss import mss


class ScreenCapture:
    """
    Captures frames from a selected screen region at a target FPS.
    Used to monitor video call windows for deepfake detection.
    """

    def __init__(self, fps: int = 10):
        self.fps = fps
        self.interval = 1.0 / fps
        self.region = None  # {"left": x, "top": y, "width": w, "height": h}
        self.is_running = False
        self.latest_frame = None
        self._thread = None
        self._lock = threading.Lock()

    def set_region(self, left: int, top: int, width: int, height: int):
        """Set the screen region to capture."""
        self.region = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }
        print(f"[ScreenCapture] Region set: {self.region}")

    def start(self):
        """Start capturing frames from the selected region."""
        if not self.region:
            raise ValueError("Region not set. Call set_region() first.")

        if self.is_running:
            return

        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[ScreenCapture] Started at {self.fps} FPS")

    def stop(self):
        """Stop capturing."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        print("[ScreenCapture] Stopped")

    def _capture_loop(self):
        """Background thread: continuously capture screen region."""
        with mss() as sct:
            while self.is_running:
                start = time.time()
                try:
                    # Capture the region
                    screenshot = sct.grab(self.region)

                    # Convert to numpy BGR (OpenCV format)
                    frame = np.array(screenshot)
                    # mss returns BGRA, convert to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    with self._lock:
                        self.latest_frame = frame

                except Exception as e:
                    print(f"[ScreenCapture] Error: {e}")

                # Maintain target FPS
                elapsed = time.time() - start
                if elapsed < self.interval:
                    time.sleep(self.interval - elapsed)

    def get_frame(self) -> np.ndarray:
        """Get the latest captured frame. Returns None if no frame yet."""
        with self._lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    @staticmethod
    def get_monitors() -> list:
        """Get list of available monitors."""
        with mss() as sct:
            return sct.monitors

    @staticmethod
    def get_full_screen_region() -> dict:
        """Get the full primary screen region."""
        with mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            return {
                "left": monitor["left"],
                "top": monitor["top"],
                "width": monitor["width"],
                "height": monitor["height"],
            }
