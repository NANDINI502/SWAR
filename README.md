# SWAR — Real-Time Multimodal Deepfake Detector

SWAR watches a video call, webcam feed, or uploaded media file and scores it in real time on two independent signals:

- **Video** — is the face on screen a deepfake (face-swap / synthetic)?
- **Audio** — is the voice a clone / AI-generated (spoof)?

## How it works

- **Video pipeline** — MTCNN face detection → face crops classified Real/Fake by a fine-tuned Vision Transformer (`ViTForImageClassification`).
- **Audio pipeline** — raw waveform windows (3s @ 16kHz) classified bonafide/spoof by a Wav2Vec2 model fine-tuned on ASVspoof2019.
- **Alert system** — aggregates a rolling window of predictions across both streams and fires an alert once sustained fake content is detected.

## Three ways to run it

| Interface | Description |
|---|---|
| `server.py` | FastAPI + WebSocket backend serving a web UI, with live-streaming (`/ws/video`, `/ws/audio`) and file-upload (`/api/analyze/video`, `/api/analyze/audio`) endpoints |
| `app.py` | Standalone Gradio UI over the same detection pipelines |
| `call_monitor.py` | Windows desktop app that screen-captures a selected window (e.g. a Zoom/Meet call) and system audio, and raises a toast alert on sustained fake content |

## Tech stack

Python, PyTorch, HuggingFace Transformers (ViT, Wav2Vec2), MTCNN, FastAPI, WebSockets, Gradio, customtkinter.

## Status

Actively developed. Core detection pipelines (video + audio) are verified working end-to-end against real model checkpoints.
