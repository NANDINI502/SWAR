"""
Alert System — Aggregates detection results into a single combined risk
score and triggers alerts. Uses rolling window consensus to avoid false
alarms (roadmap item 5, see CLAUDE_CODE_HANDOFF.md §6d).

Public API is unchanged from before item 5 — update_video(result),
update_audio(result), update_lipsync(result), get_status_summary(), reset(),
and the AlertSystem(on_alert_callback=...) constructor (callback still
receives (alert_type, message, details_dict)) all still work exactly as
call_monitor.py already calls them. What changed internally is *how* an
alert gets triggered: instead of checking each modality against its own
hard threshold independently ("alert if video crosses OR audio crosses"),
every active signal is blended into one 0-100 risk_score
(get_risk_score(), also folded into get_status_summary()) and that combined
score is what alerting is based on.
"""

import sys
import time
from collections import deque

try:
    from winotify import Notification, audio
    HAS_WINOTIFY = True
except ImportError:
    HAS_WINOTIFY = False
    print("[AlertSystem] winotify not available — toast notifications disabled")


# ─── Combined risk score settings ────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 15      # Minimum time between alerts
MIN_READINGS_FOR_SIGNAL = 3      # A signal only counts toward the combined score
                                  # once it has this many rolling readings — avoids
                                  # one noisy reading swinging the whole score.
RISK_ALERT_THRESHOLD = 50.0      # Combined risk_score (0-100) at/above this alerts.
                                  # Deliberately below a single signal's own old
                                  # "definitely fake" threshold (0.6 -> 60) so a
                                  # generator that keeps every individual signal
                                  # just under suspicion, but elevated on more than
                                  # one at once, still gets caught.
FLAGGED_SIGNAL_THRESHOLD = 0.6   # A signal's own risk above this gets named
                                  # specifically in the alert message.

# Weighted-average weights, renormalized over whichever signals are
# currently active (see _combine below) — video/audio carry the trained
# classifiers' own confidence, lipsync is corroborating evidence.
SIGNAL_WEIGHTS = {"video": 0.4, "audio": 0.4, "lipsync": 0.2}

SIGNAL_DESCRIPTIONS = {
    "video": "the face in this call appears fake",
    "audio": "the caller's voice appears AI-generated",
    "lipsync": "mouth movement doesn't match the voice",
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class AlertSystem:
    """
    Aggregates video, audio, and lip-sync detection results into a single
    combined risk score and triggers alerts when sustained overall risk is
    high — rather than requiring any one modality to independently cross
    its own hard threshold.
    """

    def __init__(self, on_alert_callback=None):
        """
        Args:
            on_alert_callback: Function called when alert triggers.
                              Receives (alert_type, message, details_dict)
        """
        self.on_alert_callback = on_alert_callback
        self.last_alert_time = 0

        # Rolling detection history (last 10 results)
        self.video_history = deque(maxlen=10)
        self.audio_history = deque(maxlen=10)
        self.lipsync_history = deque(maxlen=10)

        # Current status
        self.video_status = {"label": "IDLE", "confidence": 0.0, "fake_probability": 0.0}
        self.audio_status = {"label": "IDLE", "confidence": 0.0, "fake_probability": 0.0}
        self.lipsync_status = {"label": "IDLE", "sync_score": 0.0}
        self.alert_active = False
        self.alert_message = ""

    def update_video(self, result: dict):
        """Update with a new video detection result."""
        self.video_status = result
        self.video_history.append(result)
        self._check_alerts()

    def update_audio(self, result: dict):
        """Update with a new audio detection result."""
        self.audio_status = result
        self.audio_history.append(result)
        self._check_alerts()

    def update_lipsync(self, result: dict):
        """Update with a new lip-sync consistency result (roadmap item 1).

        `result` is whatever models.lipsync_detector.LipSyncConsistency
        .get_sync_score() returns: {"label", "sync_score", "samples"}.
        Optional third signal — callers that never call this simply never
        contribute a lipsync term to the combined risk score;
        update_video()/update_audio() behave exactly as before.
        """
        self.lipsync_status = result
        self.lipsync_history.append(result)
        self._check_alerts()

    def get_risk_score(self) -> dict:
        """
        Combine whichever signals currently have enough rolling history
        into a single 0-100 risk score.

        Returns: {"risk_score": float 0-100, "level": "IDLE"|"LOW"|"MEDIUM"|"HIGH",
                  "active_signals": {name: risk_0_to_1, ...}}
        A signal is only "active" once it has MIN_READINGS_FOR_SIGNAL
        rolling readings — with zero active signals, risk_score is 0 and
        level is "IDLE" (nothing to judge yet, not "definitely safe").
        """
        active = {}

        video_risk = self._mean_fake_probability(self.video_history)
        if video_risk is not None:
            active["video"] = video_risk

        audio_risk = self._mean_fake_probability(self.audio_history)
        if audio_risk is not None:
            active["audio"] = audio_risk

        lipsync_risk = self._mean_lipsync_risk(self.lipsync_history)
        if lipsync_risk is not None:
            active["lipsync"] = lipsync_risk

        if not active:
            return {"risk_score": 0.0, "level": "IDLE", "active_signals": {}}

        total_weight = sum(SIGNAL_WEIGHTS[name] for name in active)
        weighted_sum = sum(SIGNAL_WEIGHTS[name] * risk for name, risk in active.items())
        risk_score = 100.0 * weighted_sum / total_weight if total_weight > 0 else 0.0

        if risk_score >= RISK_ALERT_THRESHOLD:
            level = "HIGH"
        elif risk_score >= RISK_ALERT_THRESHOLD * 0.6:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {"risk_score": risk_score, "level": level, "active_signals": active}

    def _mean_fake_probability(self, history: deque):
        """Mean fake_probability over the most recent readings, or None if
        there isn't enough history yet for this signal to count."""
        if len(history) < MIN_READINGS_FOR_SIGNAL:
            return None
        recent = list(history)[-MIN_READINGS_FOR_SIGNAL:]
        return sum(r.get("fake_probability", 0.0) for r in recent) / len(recent)

    def _mean_lipsync_risk(self, history: deque):
        """Mean lipsync risk (0=in sync, 1=out of sync) over the most recent
        readings with actual data, or None if there isn't enough yet."""
        if len(history) < MIN_READINGS_FOR_SIGNAL:
            return None
        recent = [r for r in list(history)[-MIN_READINGS_FOR_SIGNAL:] if r.get("label") != "INSUFFICIENT_DATA"]
        if len(recent) < MIN_READINGS_FOR_SIGNAL:
            return None
        avg_sync_score = sum(r.get("sync_score", 0.0) for r in recent) / len(recent)
        # sync_score is roughly [-1, 1] (higher = more in sync) -> risk [0, 1]
        return _clamp01((1.0 - avg_sync_score) / 2.0)

    def _check_alerts(self):
        """Check if the combined risk score warrants an alert."""
        now = time.time()

        # Cooldown check
        if now - self.last_alert_time < ALERT_COOLDOWN_SECONDS:
            return

        risk = self.get_risk_score()

        if risk["level"] != "HIGH":
            self.alert_active = False
            self.alert_message = ""
            return

        flagged = [name for name, r in risk["active_signals"].items() if r >= FLAGGED_SIGNAL_THRESHOLD]
        named_signals = flagged or list(risk["active_signals"].keys())
        alert_type = "+".join(sorted(s.upper() for s in named_signals))
        reasons = "; ".join(SIGNAL_DESCRIPTIONS.get(s, s) for s in named_signals)
        message = f"⚠️ DEEPFAKE RISK {round(risk['risk_score'])}% — {reasons}"

        self._trigger_alert(alert_type, message, {
            "risk": risk,
            "video": self.video_status,
            "audio": self.audio_status,
            "lipsync": self.lipsync_status,
        })

    def _trigger_alert(self, alert_type: str, message: str, details: dict):
        """Fire an alert."""
        self.alert_active = True
        self.alert_message = message
        self.last_alert_time = time.time()

        self._safe_print(f"\n{'='*60}")
        self._safe_print(f"  🚨 ALERT: {message}")
        self._safe_print(f"{'='*60}\n")

        # Send Windows toast notification
        self._send_toast(alert_type, message)

        # Call UI callback
        if self.on_alert_callback:
            try:
                self.on_alert_callback(alert_type, message, details)
            except Exception as e:
                print(f"[AlertSystem] Callback error: {e}")

    def _safe_print(self, text: str):
        """Some Windows consoles default to cp1252, which can't encode the
        emoji in alert messages — degrade to '?' instead of crashing the
        alert path over a cosmetic print (this is what
        test_alert_system.py's alert-firing tests caught)."""
        try:
            print(text)
        except UnicodeEncodeError:
            encoding = sys.stdout.encoding or "ascii"
            print(text.encode(encoding, errors="replace").decode(encoding))

    def _send_toast(self, alert_type: str, message: str):
        """Send a Windows toast notification."""
        if not HAS_WINOTIFY:
            return

        try:
            toast = Notification(
                app_id="Deepfake Detector",
                title="🚨 Deepfake Detected!",
                msg=message,
                duration="long",
            )
            toast.set_audio(audio.LoopingAlarm, loop=False)
            toast.show()
        except Exception as e:
            print(f"[AlertSystem] Toast error: {e}")

    def get_status_summary(self) -> dict:
        """Get current status for UI display."""
        return {
            "video": self.video_status,
            "audio": self.audio_status,
            "lipsync": self.lipsync_status,
            "risk": self.get_risk_score(),
            "alert_active": self.alert_active,
            "alert_message": self.alert_message,
        }

    def reset(self):
        """Clear all history and alerts."""
        self.video_history.clear()
        self.audio_history.clear()
        self.lipsync_history.clear()
        self.video_status = {"label": "IDLE", "confidence": 0.0, "fake_probability": 0.0}
        self.audio_status = {"label": "IDLE", "confidence": 0.0, "fake_probability": 0.0}
        self.lipsync_status = {"label": "IDLE", "sync_score": 0.0}
        self.alert_active = False
        self.alert_message = ""
