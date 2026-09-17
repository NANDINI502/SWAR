"""
Real-Time Deepfake Detection System
====================================
Gradio-based UI for detecting deepfake video (faces) and audio (voice) in real-time.

Usage:
    python app.py

Features:
    - Live webcam feed with face deepfake detection
    - Live microphone with voice deepfake detection
    - Upload video/audio files for analysis
"""

import os
import sys
import cv2
import time
import numpy as np
import gradio as gr
import torch

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, GRADIO_SERVER_PORT, GRADIO_SHARE,
    VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT,
    SAMPLE_RATE, AUDIO_WINDOW_SECONDS,
)
from pipeline.video_pipeline import VideoPipeline
from pipeline.audio_pipeline import AudioPipeline

# ─── Global Pipelines ───────────────────────────────────────────────────────────
print("=" * 60)
print("  Real-Time Deepfake Detection System")
print(f"  Device: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("=" * 60)

video_pipeline = VideoPipeline(device=DEVICE)
audio_pipeline = AudioPipeline(device=DEVICE)


# ─── Video Detection (Webcam) ───────────────────────────────────────────────────

def process_webcam_frame(frame):
    """Process a single webcam frame from Gradio's Image component."""
    if frame is None:
        return None, "No frame received"

    # Gradio sends RGB, OpenCV expects BGR
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    bgr_frame = cv2.resize(bgr_frame, (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT))

    annotated, result = video_pipeline.process_frame(bgr_frame)

    # Convert back to RGB for Gradio display
    rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Format result text
    label = result.get("label", "DETECTING...")
    confidence = result.get("confidence", 0.0)
    fake_prob = result.get("fake_probability", 0.0)

    status_text = f"### 🎭 Video Detection\n"
    if label == "FAKE":
        status_text += f"**⚠️ FAKE DETECTED** — Confidence: {confidence:.1%}\n\n"
    elif label == "REAL":
        status_text += f"**✅ REAL** — Confidence: {confidence:.1%}\n\n"
    else:
        status_text += f"**🔍 {label}**\n\n"

    status_text += f"Fake Probability: {fake_prob:.1%} | FPS: {video_pipeline.fps:.0f}"

    return rgb_annotated, status_text


# ─── Video Detection (File Upload) ──────────────────────────────────────────────

def process_uploaded_video(video_path):
    """Process an uploaded video file."""
    if video_path is None:
        return "No video uploaded"

    results = []
    frame_count = 0
    fake_count = 0

    for annotated, result in video_pipeline.process_uploaded_video(video_path):
        frame_count += 1
        if result.get("label") == "FAKE":
            fake_count += 1
        results.append(result)

    total_analyzed = len([r for r in results if r.get("label") in ["REAL", "FAKE"]])
    if total_analyzed == 0:
        return "No faces detected in the video."

    fake_pct = fake_count / total_analyzed * 100
    avg_fake_prob = np.mean([r["fake_probability"] for r in results if "fake_probability" in r])

    report = f"### 📊 Video Analysis Report\n\n"
    report += f"- **Frames analyzed**: {total_analyzed}\n"
    report += f"- **Fake frames**: {fake_count} ({fake_pct:.1f}%)\n"
    report += f"- **Average fake probability**: {avg_fake_prob:.1%}\n\n"

    if fake_pct > 50:
        report += "**⚠️ Verdict: This video is likely a DEEPFAKE**"
    elif fake_pct > 20:
        report += "**⚡ Verdict: SUSPICIOUS — some frames appear manipulated**"
    else:
        report += "**✅ Verdict: This video appears AUTHENTIC**"

    return report


# ─── Audio Detection (Microphone) ───────────────────────────────────────────────

def process_audio_chunk(audio):
    """Process audio from Gradio's Audio component."""
    if audio is None:
        return "Waiting for audio..."

    sr, data = audio

    # Convert to mono float32 if needed
    if data.ndim > 1:
        data = data.mean(axis=1)

    data = data.astype(np.float32)

    # Normalize to [-1, 1] range
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val

    # Resample if needed
    if sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Need at least the window size
    min_samples = int(AUDIO_WINDOW_SECONDS * SAMPLE_RATE)
    if len(data) < min_samples:
        data = np.pad(data, (0, min_samples - len(data)))

    # Analyze in windows
    results = audio_pipeline.process_audio_file_from_array(data)

    if not results:
        result = audio_pipeline.detector.predict(data[:min_samples])
        results = [result]

    # Aggregate
    avg_fake_prob = np.mean([r["fake_probability"] for r in results])
    label = "FAKE" if avg_fake_prob > 0.5 else "REAL"

    status_text = f"### 🎤 Voice Detection\n\n"
    if label == "FAKE":
        status_text += f"**⚠️ FAKE VOICE DETECTED** — Score: {avg_fake_prob:.1%}\n\n"
    else:
        status_text += f"**✅ REAL VOICE** — Score: {1 - avg_fake_prob:.1%}\n\n"

    status_text += f"Windows analyzed: {len(results)} | "
    status_text += f"Avg fake probability: {avg_fake_prob:.1%}"

    return status_text


# ─── Audio Detection (File Upload) ──────────────────────────────────────────────

def process_uploaded_audio(audio_path):
    """Process an uploaded audio file."""
    if audio_path is None:
        return "No audio uploaded"

    results = audio_pipeline.process_audio_file(audio_path)

    if not results:
        return "Could not analyze audio file. It may be too short."

    avg_fake_prob = np.mean([r["fake_probability"] for r in results])
    fake_windows = sum(1 for r in results if r["label"] == "FAKE")

    report = f"### 📊 Audio Analysis Report\n\n"
    report += f"- **Windows analyzed**: {len(results)}\n"
    report += f"- **Fake windows**: {fake_windows}/{len(results)}\n"
    report += f"- **Average fake probability**: {avg_fake_prob:.1%}\n\n"

    # Timeline
    report += "**Timeline:**\n\n"
    for r in results:
        emoji = "🔴" if r["label"] == "FAKE" else "🟢"
        report += f"  {emoji} [{r.get('time_start', '?')}s - {r.get('time_end', '?')}s] "
        report += f"{r['label']} ({r['fake_probability']:.1%})\n\n"

    if avg_fake_prob > 0.5:
        report += "\n**⚠️ Verdict: This audio is likely FAKE / AI-generated**"
    else:
        report += "\n**✅ Verdict: This audio appears AUTHENTIC**"

    return report


# ─── Build Gradio UI ────────────────────────────────────────────────────────────

CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

.status-real {
    color: #22c55e;
    font-weight: bold;
}

.status-fake {
    color: #ef4444;
    font-weight: bold;
}

h1 {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem !important;
    margin-bottom: 0.5rem !important;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 1.5rem;
}
"""

def build_ui():
    """Build the Gradio interface."""
    with gr.Blocks(
        title="Real-Time Deepfake Detector",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
        ),
    ) as demo:

        gr.Markdown("# 🛡️ Real-Time Deepfake Detector")
        gr.Markdown(
            "<p class='subtitle'>Detect deepfake videos and AI-generated voices in real-time</p>"
        )

        with gr.Tabs():
            # ─── Tab 1: Live Detection ──────────────────────────────────────
            with gr.Tab("🔴 Live Detection", id="live"):
                gr.Markdown("### Webcam & Microphone — Real-time Analysis")
                gr.Markdown(
                    "> Point your webcam at a video call or screen to detect deepfakes. "
                    "Speak into your microphone to detect voice clones."
                )

                with gr.Row():
                    # Video column
                    with gr.Column(scale=2):
                        webcam_input = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            label="📷 Webcam Feed",
                        )
                        video_output = gr.Image(label="🎭 Detection Result")
                        video_status = gr.Markdown("### 🎭 Video Detection\n\nWaiting for webcam...")

                    # Audio column
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            label="🎤 Microphone",
                            streaming=False,
                        )
                        audio_status = gr.Markdown("### 🎤 Voice Detection\n\nWaiting for audio...")

                # Wire up webcam streaming
                webcam_input.stream(
                    fn=process_webcam_frame,
                    inputs=[webcam_input],
                    outputs=[video_output, video_status],
                    show_progress="hidden",
                )

                # Wire up audio
                audio_input.change(
                    fn=process_audio_chunk,
                    inputs=[audio_input],
                    outputs=[audio_status],
                )

            # ─── Tab 2: Upload Analysis ─────────────────────────────────────
            with gr.Tab("📁 Upload Analysis", id="upload"):
                gr.Markdown("### Upload a video or audio file for deepfake analysis")

                with gr.Row():
                    with gr.Column():
                        video_upload = gr.Video(label="📹 Upload Video")
                        video_analyze_btn = gr.Button(
                            "🔍 Analyze Video", variant="primary"
                        )
                        video_report = gr.Markdown("Upload a video to analyze")

                    with gr.Column():
                        audio_upload = gr.Audio(
                            type="filepath",
                            label="🔊 Upload Audio",
                        )
                        audio_analyze_btn = gr.Button(
                            "🔍 Analyze Audio", variant="primary"
                        )
                        audio_report = gr.Markdown("Upload an audio file to analyze")

                video_analyze_btn.click(
                    fn=process_uploaded_video,
                    inputs=[video_upload],
                    outputs=[video_report],
                )

                audio_analyze_btn.click(
                    fn=process_uploaded_audio,
                    inputs=[audio_upload],
                    outputs=[audio_report],
                )

            # ─── Tab 3: About ───────────────────────────────────────────────
            with gr.Tab("ℹ️ About", id="about"):
                video_status = (
                    "✅ Loaded from weights/video_model"
                    if video_pipeline.detector.ready
                    else "⚠️ Not loaded — see the [VideoDetector] warning in the console, then check weights/video_model/"
                )
                audio_status = (
                    "✅ Loaded from weights/audio_model"
                    if audio_pipeline.detector.ready
                    else "⚠️ Not loaded — see the [AudioDetector] warning in the console, then check weights/audio_model/"
                )

                gr.Markdown("""
### How It Works

**Video Detection:**
1. MTCNN detects faces in each frame
2. Face crops are classified by a fine-tuned ViT (Vision Transformer)
3. The model outputs a probability score (REAL vs FAKE)

**Audio Detection:**
1. Audio is captured in sliding windows (3 seconds)
2. The raw waveform is fed directly into a fine-tuned Wav2Vec2 classifier
3. Wav2Vec2 outputs bonafide/spoof, mapped to REAL/FAKE

### Model Status

| Model | Status |
|---|---|
""" + f"| Video (ViT) | {video_status} |\n"
    + f"| Audio (Wav2Vec2) | {audio_status} |\n" + """
### Tips for Best Results
- Ensure good lighting for face detection
- Speak clearly into the microphone
- If either status above shows ⚠️, re-run the matching script in `training/` and extract its output into the `weights/` folder named in the warning

### Hardware
""" + f"- **Device**: {DEVICE.upper()}\n" +
    (f"- **GPU**: {torch.cuda.get_device_name(0)}\n" if DEVICE == "cuda" else "") +
    f"- **PyTorch version**: {torch.__version__}\n"
                )

    return demo


# ─── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_port=GRADIO_SERVER_PORT,
        share=GRADIO_SHARE,
        show_error=True,
    )
