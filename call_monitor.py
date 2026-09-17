"""
Real-Time Call Deepfake Detector — Desktop App
===============================================
Monitors video/voice calls in real-time and alerts when deepfakes are detected.

Usage:
    python call_monitor.py

Features:
    - Screen region capture (monitor any video call window)
    - System audio capture (WASAPI loopback — captures caller's voice)
    - Real-time deepfake detection with alert system
    - Windows toast notifications on detection
"""

import os
import sys
import time
import threading
import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk

import customtkinter as ctk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE, SAMPLE_RATE, AUDIO_WINDOW_SECONDS
from models.face_extractor import FaceExtractor
from models.video_detector import VideoDetectorInference
from models.audio_detector import AudioDetectorInference
from capture.screen_capture import ScreenCapture
from capture.audio_capture import SystemAudioCapture
from alert_system import AlertSystem

import torch

# ─── Theme ───────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class RegionSelector(tk.Toplevel):
    """
    Transparent overlay for selecting the screen region to monitor.
    Uses native tkinter (not CustomTkinter) for reliable canvas + transparency.
    """

    def __init__(self, parent, on_region_selected):
        super().__init__(parent)
        self.on_region_selected = on_region_selected

        # Make fullscreen transparent overlay
        self.attributes("-fullscreen", True)
        self.attributes("-alpha", 0.3)
        self.attributes("-topmost", True)
        self.configure(bg="gray10")

        # Native tkinter canvas for drawing
        self.canvas = tk.Canvas(self, bg="gray10", highlightthickness=0, cursor="cross")
        self.canvas.pack(fill="both", expand=True)

        # Instruction text on canvas
        self.canvas.create_text(
            self.winfo_screenwidth() // 2, 50,
            text="Drag to select the video call area, then release  |  Press ESC to cancel",
            font=("Segoe UI", 18, "bold"),
            fill="white",
        )

        # Drag state
        self.start_x = 0
        self.start_y = 0
        self.rect_id = None

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Escape>", lambda e: self.destroy())

    def _on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)

    def _on_drag(self, event):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline="lime", width=3, dash=(6, 4),
        )

    def _on_release(self, event):
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        w = x2 - x1
        h = y2 - y1

        if w > 50 and h > 50:
            self.on_region_selected(x1, y1, w, h)

        self.destroy()


class CallMonitorApp(ctk.CTk):
    """Main desktop app for real-time call deepfake detection."""

    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Deepfake Call Monitor")
        self.geometry("520x720")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Core components
        self.screen_capture = ScreenCapture(fps=10)
        self.audio_capture = SystemAudioCapture(sample_rate=SAMPLE_RATE)
        self.face_extractor = FaceExtractor(device=DEVICE)
        self.video_detector = VideoDetectorInference(device=DEVICE)
        self.audio_detector = AudioDetectorInference(device=DEVICE)
        self.alert_system = AlertSystem(on_alert_callback=self._on_alert)

        # State
        self.is_monitoring = False
        self._monitor_thread = None

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the desktop UI."""
        # ─── Header ──────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(15, 5))

        ctk.CTkLabel(
            header,
            text="Deepfake Call Monitor",
            font=("Segoe UI", 24, "bold"),
        ).pack()

        ctk.CTkLabel(
            header,
            text=f"Device: {DEVICE.upper()}" + (f" — {torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else ""),
            font=("Segoe UI", 11),
            text_color="gray",
        ).pack()

        # ─── Alert Banner ────────────────────────────────────────────────
        self.alert_frame = ctk.CTkFrame(self, fg_color="#dc2626", corner_radius=10, height=0)
        self.alert_frame.pack(fill="x", padx=20, pady=(5, 0))
        self.alert_frame.pack_forget()  # Hidden initially

        self.alert_label = ctk.CTkLabel(
            self.alert_frame,
            text="",
            font=("Segoe UI", 14, "bold"),
            text_color="white",
            wraplength=460,
        )
        self.alert_label.pack(padx=15, pady=10)

        # ─── Control Buttons ─────────────────────────────────────────────
        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.pack(fill="x", padx=20, pady=10)

        self.region_btn = ctk.CTkButton(
            controls,
            text="Select Call Window",
            command=self._select_region,
            font=("Segoe UI", 14),
            height=40,
            fg_color="#6366f1",
            hover_color="#4f46e5",
        )
        self.region_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.monitor_btn = ctk.CTkButton(
            controls,
            text="Start Monitoring",
            command=self._toggle_monitoring,
            font=("Segoe UI", 14, "bold"),
            height=40,
            fg_color="#22c55e",
            hover_color="#16a34a",
            state="disabled",
        )
        self.monitor_btn.pack(side="right", expand=True, fill="x", padx=(5, 0))

        # ─── Region Info ─────────────────────────────────────────────────
        self.region_label = ctk.CTkLabel(
            self,
            text="No region selected — click 'Select Call Window' to start",
            font=("Segoe UI", 11),
            text_color="#9ca3af",
        )
        self.region_label.pack(padx=20, pady=(0, 5))

        # ─── Preview Frame ───────────────────────────────────────────────
        preview_container = ctk.CTkFrame(self, fg_color="#1e1e2e", corner_radius=10)
        preview_container.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            preview_container,
            text="Video Preview",
            font=("Segoe UI", 12, "bold"),
        ).pack(padx=10, pady=(8, 0))

        self.preview_label = ctk.CTkLabel(
            preview_container,
            text="Waiting for capture...",
            width=460,
            height=260,
            fg_color="#111827",
            corner_radius=8,
        )
        self.preview_label.pack(padx=10, pady=10)

        # ─── Detection Status ────────────────────────────────────────────
        status_frame = ctk.CTkFrame(self, fg_color="#1e1e2e", corner_radius=10)
        status_frame.pack(fill="x", padx=20, pady=5)

        # Video status
        video_row = ctk.CTkFrame(status_frame, fg_color="transparent")
        video_row.pack(fill="x", padx=15, pady=(10, 5))

        ctk.CTkLabel(video_row, text="Video:", font=("Segoe UI", 13, "bold")).pack(side="left")
        self.video_label = ctk.CTkLabel(
            video_row, text="IDLE", font=("Segoe UI", 13, "bold"), text_color="#9ca3af"
        )
        self.video_label.pack(side="left", padx=10)

        self.video_bar = ctk.CTkProgressBar(video_row, width=200, height=14)
        self.video_bar.pack(side="right")
        self.video_bar.set(0)

        # Audio status
        audio_row = ctk.CTkFrame(status_frame, fg_color="transparent")
        audio_row.pack(fill="x", padx=15, pady=(0, 5))

        ctk.CTkLabel(audio_row, text="Audio:", font=("Segoe UI", 13, "bold")).pack(side="left")
        self.audio_label = ctk.CTkLabel(
            audio_row, text="IDLE", font=("Segoe UI", 13, "bold"), text_color="#9ca3af"
        )
        self.audio_label.pack(side="left", padx=10)

        self.audio_bar = ctk.CTkProgressBar(audio_row, width=200, height=14)
        self.audio_bar.pack(side="right")
        self.audio_bar.set(0)

        # Audio level
        level_row = ctk.CTkFrame(status_frame, fg_color="transparent")
        level_row.pack(fill="x", padx=15, pady=(0, 10))

        ctk.CTkLabel(level_row, text="Level:", font=("Segoe UI", 11)).pack(side="left")
        self.level_bar = ctk.CTkProgressBar(level_row, width=200, height=8, progress_color="#06b6d4")
        self.level_bar.pack(side="right")
        self.level_bar.set(0)

        # ─── Audio Device Selection ──────────────────────────────────────
        audio_device_frame = ctk.CTkFrame(self, fg_color="#1e1e2e", corner_radius=10)
        audio_device_frame.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            audio_device_frame,
            text="Audio Source",
            font=("Segoe UI", 12, "bold"),
        ).pack(padx=10, pady=(8, 0))

        device_row = ctk.CTkFrame(audio_device_frame, fg_color="transparent")
        device_row.pack(fill="x", padx=10, pady=(5, 10))

        self.audio_mode = ctk.CTkSegmentedButton(
            device_row,
            values=["System Audio (Loopback)", "Microphone"],
            command=self._on_audio_mode_change,
            font=("Segoe UI", 11),
        )
        self.audio_mode.set("System Audio (Loopback)")
        self.audio_mode.pack(fill="x")

        # ─── Footer ─────────────────────────────────────────────────────
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=20, pady=(5, 15))

        self.status_line = ctk.CTkLabel(
            footer,
            text="Ready — select a region to begin",
            font=("Segoe UI", 11),
            text_color="#6b7280",
        )
        self.status_line.pack()

    # ─── Region Selection ────────────────────────────────────────────────────

    def _select_region(self):
        """Open the transparent overlay for region selection."""
        self.iconify()  # Minimize while selecting
        self.after(300, lambda: RegionSelector(self, self._on_region_selected))

    def _on_region_selected(self, x, y, w, h):
        """Handle region selection."""
        self.deiconify()  # Restore window
        self.screen_capture.set_region(x, y, w, h)
        self.region_label.configure(
            text=f"Region: ({x}, {y}) - {w}x{h} px",
            text_color="#22c55e",
        )
        self.monitor_btn.configure(state="normal")
        self.status_line.configure(text="Region selected — click 'Start Monitoring' to begin")

    # ─── Monitoring Control ──────────────────────────────────────────────────

    def _toggle_monitoring(self):
        """Start or stop monitoring."""
        if self.is_monitoring:
            self._stop_monitoring()
        else:
            self._start_monitoring()

    def _start_monitoring(self):
        """Start the monitoring engine."""
        self.is_monitoring = True
        self.monitor_btn.configure(
            text="Stop Monitoring",
            fg_color="#ef4444",
            hover_color="#dc2626",
        )
        self.region_btn.configure(state="disabled")
        self.alert_system.reset()

        # Start capture
        self.screen_capture.start()
        use_loopback = self.audio_mode.get() == "System Audio (Loopback)"
        self.audio_capture.start(use_loopback=use_loopback)

        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        # Start UI update loop
        self._update_ui()

        self.status_line.configure(text="[ACTIVE] Monitoring -- watching for deepfakes...")

    def _stop_monitoring(self):
        """Stop the monitoring engine."""
        self.is_monitoring = False
        self.screen_capture.stop()
        self.audio_capture.stop()

        self.monitor_btn.configure(
            text="Start Monitoring",
            fg_color="#22c55e",
            hover_color="#16a34a",
        )
        self.region_btn.configure(state="normal")
        self.status_line.configure(text="Monitoring stopped")

        # Hide alert
        self.alert_frame.pack_forget()

    # ─── Monitoring Loop ─────────────────────────────────────────────────────

    def _monitor_loop(self):
        """Background thread: run detection on captured data."""
        frame_count = 0

        while self.is_monitoring:
            try:
                # ── Video Detection ──────────────────────────────────────
                frame = self.screen_capture.get_frame()
                if frame is not None:
                    frame_count += 1

                    # Process every 3rd frame
                    if frame_count % 3 == 0:
                        faces = self.face_extractor.extract_faces(frame)
                        if faces:
                            best_face = max(faces, key=lambda f: f["confidence"])
                            result = self.video_detector.predict(best_face["face"])
                            self.alert_system.update_video(result)
                        else:
                            self.alert_system.update_video({
                                "label": "NO FACE",
                                "confidence": 0.0,
                                "fake_probability": 0.0,
                            })

                # ── Audio Detection ──────────────────────────────────────
                audio_window = self.audio_capture.get_latest_window()
                if audio_window is not None:
                    result = self.audio_detector.predict(audio_window)
                    self.alert_system.update_audio(result)

                time.sleep(0.05)

            except Exception as e:
                print(f"[Monitor] Error: {e}")
                time.sleep(0.2)

    # ─── UI Updates ──────────────────────────────────────────────────────────

    def _update_ui(self):
        """Periodically update the UI with latest results."""
        if not self.is_monitoring:
            return

        status = self.alert_system.get_status_summary()

        # Update video status
        v = status["video"]
        v_label = v.get("label", "IDLE")
        v_prob = v.get("fake_probability", 0.0)
        self.video_label.configure(
            text=f"{v_label} ({v_prob:.0%})",
            text_color="#ef4444" if v_label == "FAKE" else "#22c55e" if v_label == "REAL" else "#9ca3af",
        )
        self.video_bar.set(v_prob)
        self.video_bar.configure(
            progress_color="#ef4444" if v_prob > 0.5 else "#22c55e"
        )

        # Update audio status
        a = status["audio"]
        a_label = a.get("label", "IDLE")
        a_prob = a.get("fake_probability", 0.0)
        self.audio_label.configure(
            text=f"{a_label} ({a_prob:.0%})",
            text_color="#ef4444" if a_label == "FAKE" else "#22c55e" if a_label == "REAL" else "#9ca3af",
        )
        self.audio_bar.set(a_prob)
        self.audio_bar.configure(
            progress_color="#ef4444" if a_prob > 0.5 else "#22c55e"
        )

        # Update audio level
        level = self.audio_capture.get_audio_level()
        self.level_bar.set(min(level * 5, 1.0))

        # Update alert banner
        if status["alert_active"]:
            self.alert_label.configure(text=status["alert_message"])
            self.alert_frame.pack(fill="x", padx=20, pady=(5, 0), after=self.winfo_children()[0])
        else:
            self.alert_frame.pack_forget()

        # Update preview
        frame = self.screen_capture.get_frame()
        if frame is not None:
            self._update_preview(frame)

        # Schedule next update
        self.after(200, self._update_ui)

    def _update_preview(self, frame: np.ndarray):
        """Update the video preview with the captured frame."""
        try:
            # Resize to preview size
            h, w = frame.shape[:2]
            target_w, target_h = 460, 260
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_resized = cv2.resize(frame, (new_w, new_h))

            # Convert BGR → RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Convert to CTk image
            pil_image = Image.fromarray(frame_rgb)
            ctk_image = ctk.CTkImage(pil_image, size=(new_w, new_h))
            self.preview_label.configure(image=ctk_image, text="")
            self.preview_label._ctk_image = ctk_image  # Prevent garbage collection

        except Exception:
            pass

    # ─── Callbacks ───────────────────────────────────────────────────────────

    def _on_alert(self, alert_type, message, details):
        """Called by AlertSystem when a deepfake is detected."""
        # UI update happens in _update_ui via status check
        pass

    def _on_audio_mode_change(self, value):
        """Handle audio source change."""
        if self.is_monitoring:
            # Restart audio capture with new mode
            self.audio_capture.stop()
            use_loopback = value == "System Audio (Loopback)"
            self.audio_capture.start(use_loopback=use_loopback)

    def _on_close(self):
        """Clean shutdown."""
        if self.is_monitoring:
            self._stop_monitoring()
        self.destroy()


# ─── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Deepfake Call Monitor")
    print(f"  Device: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    print()

    app = CallMonitorApp()
    app.mainloop()
