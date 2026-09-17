"""
Smoke test for alert_system.py's combined risk score (roadmap item 5).

Checks the specific behavior item 5 exists for: a generator that keeps
*every individual signal* just under its own old hard threshold (0.6) should
still be caught once enough of them are simultaneously elevated, because the
combined score blends them instead of requiring any one to independently
cross a line. Also checks the old single-strong-signal case (one modality
alone, clearly fake) still alerts — the ensemble should degrade gracefully,
not become *less* sensitive than the old per-modality checks it replaced.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alert_system import AlertSystem, MIN_READINGS_FOR_SIGNAL, ALERT_COOLDOWN_SECONDS


def video_result(fake_probability: float) -> dict:
    label = "FAKE" if fake_probability > 0.5 else "REAL"
    return {"label": label, "confidence": fake_probability, "fake_probability": fake_probability}


def audio_result(fake_probability: float) -> dict:
    label = "FAKE" if fake_probability > 0.5 else "REAL"
    return {"label": label, "confidence": fake_probability, "fake_probability": fake_probability}


def lipsync_result(sync_score: float) -> dict:
    label = "SYNCED" if sync_score >= 0.3 else "OUT_OF_SYNC"
    return {"label": label, "sync_score": sync_score, "samples": 40}


print("=" * 60)
print("Test: no readings yet -> IDLE, no alert")
print("=" * 60)
try:
    alerts = AlertSystem()
    risk = alerts.get_risk_score()
    print(f"Risk with no data: {risk}")
    assert risk["level"] == "IDLE" and risk["risk_score"] == 0.0
    assert alerts.alert_active is False
    print("[OK]")
except Exception as e:
    print(f"[FAIL] {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("Test: multiple sub-threshold signals combine into a HIGH-risk alert")
print("=" * 60)
try:
    received = []
    alerts = AlertSystem(on_alert_callback=lambda *args: received.append(args))

    # Each signal alone stays under the OLD per-modality hard threshold
    # (0.6) that item 5 replaced — video and audio both sit at 0.55.
    for _ in range(MIN_READINGS_FOR_SIGNAL):
        alerts.update_video(video_result(0.55))
        alerts.update_audio(audio_result(0.55))

    risk = alerts.get_risk_score()
    print(f"Risk after 3x video=0.55, audio=0.55: {risk}")

    assert "video" in risk["active_signals"] and "audio" in risk["active_signals"]
    assert abs(risk["active_signals"]["video"] - 0.55) < 1e-6
    assert risk["level"] == "HIGH", f"expected combined score to cross into HIGH, got {risk}"
    assert alerts.alert_active is True
    assert len(received) == 1, "on_alert_callback should have fired exactly once"
    alert_type, message, details = received[0]
    print(f"Callback received: alert_type={alert_type!r}, message length={len(message)}")
    assert isinstance(alert_type, str) and isinstance(message, str) and isinstance(details, dict)
    assert "risk" in details

    print("[OK] Two individually sub-threshold signals combined into a HIGH-risk alert")
except Exception as e:
    print(f"[FAIL] {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("Test: a single, clearly-fake signal alone still alerts (renormalized weight)")
print("=" * 60)
try:
    alerts = AlertSystem()
    for _ in range(MIN_READINGS_FOR_SIGNAL):
        alerts.update_video(video_result(0.95))

    risk = alerts.get_risk_score()
    print(f"Risk after 3x video=0.95 alone: {risk}")
    assert list(risk["active_signals"].keys()) == ["video"]
    assert risk["level"] == "HIGH"
    assert alerts.alert_active is True
    print("[OK] Single strongly-fake signal still triggers on its own")
except Exception as e:
    print(f"[FAIL] {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("Test: all-real, in-sync signals stay LOW risk, no alert")
print("=" * 60)
try:
    alerts = AlertSystem()
    for _ in range(MIN_READINGS_FOR_SIGNAL):
        alerts.update_video(video_result(0.05))
        alerts.update_audio(audio_result(0.05))
        alerts.update_lipsync(lipsync_result(0.9))

    risk = alerts.get_risk_score()
    print(f"Risk with all-real, in-sync signals: {risk}")
    assert risk["level"] in ("LOW",)
    assert alerts.alert_active is False
    print("[OK]")
except Exception as e:
    print(f"[FAIL] {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("Test: lipsync INSUFFICIENT_DATA readings don't count as an active signal")
print("=" * 60)
try:
    alerts = AlertSystem()
    for _ in range(MIN_READINGS_FOR_SIGNAL):
        alerts.update_lipsync({"label": "INSUFFICIENT_DATA", "sync_score": 0.0, "samples": 0})

    risk = alerts.get_risk_score()
    print(f"Risk with only INSUFFICIENT_DATA lipsync readings: {risk}")
    assert "lipsync" not in risk["active_signals"]
    assert risk["level"] == "IDLE"
    print("[OK] INSUFFICIENT_DATA correctly excluded rather than treated as risk-free or risky")
except Exception as e:
    print(f"[FAIL] {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("Test: get_status_summary() keeps existing keys call_monitor.py reads")
print("=" * 60)
try:
    alerts = AlertSystem()
    status = alerts.get_status_summary()
    for key in ("video", "audio", "lipsync", "alert_active", "alert_message"):
        assert key in status, f"missing expected key: {key}"
    assert "risk" in status  # new, additive
    print(f"Status summary keys: {sorted(status.keys())}")
    print("[OK] Backward-compatible status keys preserved, risk score added")
except Exception as e:
    print(f"[FAIL] {e}")
    traceback.print_exc()

print("\nDONE")
