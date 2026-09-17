# SWAR — `deepfake_detector` Project Handoff

**Purpose of this file:** full context for whoever (human or Claude Code) picks up work on this project next, without needing to re-derive it from scratch. Read this entire file before touching code.

Location on disk: `C:\Users\nandi\OneDrive\Desktop\swar\deepfake_detector\`

---

## 1. What this project is

A real-time multimodal deepfake detector. It watches a video call / webcam / uploaded file and, in real time, scores:
- **Video**: is the face on screen a deepfake (face-swap / synthetic)?
- **Audio**: is the voice a clone / AI-generated (spoof)?

Three ways to run it:
- `server.py` — FastAPI + WebSocket backend, serves `static/index.html` (the "Authix" web UI). Endpoints: `/ws/video`, `/ws/audio` (live streaming), `/api/analyze/video`, `/api/analyze/audio` (file upload).
- `app.py` — Gradio UI, same detection pipelines, standalone.
- `call_monitor.py` — Windows desktop app (customtkinter) that screen-captures a selected region (e.g. a Zoom/Meet window) + captures system audio via WASAPI loopback, and pops a toast alert when sustained fake content is detected.

All three sit on top of the same two pipelines:
- `pipeline/video_pipeline.py` → `models/face_extractor.py` (MTCNN face detection) → `models/video_detector.py` (classifies each face crop REAL/FAKE)
- `pipeline/audio_pipeline.py` → `models/audio_detector.py` (classifies 3-second audio windows REAL/FAKE)

`alert_system.py` aggregates a rolling window of results and fires an alert (+ Windows toast) when N consecutive fake detections happen on either/both streams.

---

## 2. Current architecture (as of this handoff — verified working)

**Video model**: `ViTForImageClassification` (HuggingFace `transformers`), loaded from `weights/video_model/` (local checkpoint: `config.json`, `model.safetensors`, `preprocessor_config.json`). `id2label = {0: "Real", 1: "Fake"}`. Classifies **one face crop at a time** (no temporal/sequence modeling — see §7 item 1 for why this matters).

**Audio model**: `Wav2Vec2ForSequenceClassification` (HuggingFace `transformers`), loaded from `weights/audio_model/` (local checkpoint: `config.json`, `pytorch_model.bin`, `preprocessor_config.json`). `id2label = {0: "bonafide", 1: "spoof"}` — fine-tuned on ASVspoof2019. Classifies raw waveform windows (3 seconds @ 16kHz, `AUDIO_WINDOW_SECONDS`/`SAMPLE_RATE` in `config.py`).

Both detector classes (`models/audio_detector.py::AudioDetectorInference`, `models/video_detector.py::VideoDetectorInference`) expose the same contract — **do not break this when adding features**:

```python
detector = AudioDetectorInference(device="cpu")   # or VideoDetectorInference
detector.ready  # bool — False if weights/transformers missing
result = detector.predict(input)
# result = {"label": "REAL"|"FAKE"|"ERROR"|"PRED ERROR", "confidence": float 0-1, "fake_probability": float 0-1}
```

`VideoDetectorInference.predict()` takes a **raw RGB uint8 numpy array** (H,W,3) or PIL Image — NOT pre-normalized. `FaceExtractor.extract_faces()` returns crops in exactly this format (`{"face": np.ndarray, "box": (x1,y1,x2,y2), "confidence": float}`).

`AudioDetectorInference.predict(waveform, sr=None)` takes a 1D float32 numpy array at any sample rate (resampled internally to the model's rate).

Hardware: her laptop has an NVIDIA GPU — `torch` was already installed as `torchaudio 2.5.1+cu121` before this session touched anything, so `config.py`'s `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` will likely resolve to `"cuda"` at runtime. Keep any new code CUDA-optional (fall back to CPU) but feel free to use the GPU for anything heavier.

---

## 3. Bug history — what was broken and why (2026-09-17 session)

**Do not reintroduce these.** Full root-cause writeup:

### Bug A — Audio "hallucination" (root cause of the original complaint)
`models/audio_detector.py` originally instantiated a custom from-scratch `AudioDeepfakeCNN` (`models/custom_audio_net.py`) and loaded weights from `weights/audio_cnn.pth`. **That file never existed** — no script in the repo ever produces it (the from-scratch training script `training/train_custom_audio.py` would produce it, but was apparently never run to completion). The actual trained checkpoint that exists on disk, `weights/audio_model/` (a fine-tuned Wav2Vec2, downloaded via `training/train_audio_kaggle.py`, claimed 99.98% accuracy on ASVspoof2019), is for a **completely different architecture** that the old code never loaded. Because the `.pth` path didn't exist, the code silently fell back to a randomly-initialized `AudioDeepfakeCNN` and ran inference on untrained weights — producing labels/confidences with no relationship to the actual input. That is what "hallucinating" meant.

### Bug B — Video detection was 100% dead (not just untrained — never even ran)
`models/face_extractor.py` returned an **ImageNet-normalized `torch.Tensor`** from `extract_faces()`. The old `models/video_detector.py::predict()` only accepted `np.ndarray` or `PIL.Image` — anything else (including a `torch.Tensor`) hit an `else: return {"label": "UNKNOWN FORMAT", ...}` branch. So on every single call, in every entry point (`server.py`'s `/ws/video`, `app.py`'s webcam handler, `call_monitor.py`'s monitor loop), video prediction short-circuited to `"UNKNOWN FORMAT"` before the model ever ran. Same underlying weights problem as Bug A also applied here (`weights/video_lstm.pth` never existed; the real checkpoint is the HF ViT in `weights/video_model/`).

### The fix
- Rewrote `models/audio_detector.py` and `models/video_detector.py` to load the **actual existing checkpoints** (Wav2Vec2 / ViT via `transformers`, from `weights/audio_model/` and `weights/video_model/`) instead of the nonexistent custom-net `.pth` files. No retraining needed — these checkpoints were already trained and verified (see `test_output.txt`, a log of `test_models.py` loading them successfully via raw HF `pipeline()` calls).
- Changed `models/face_extractor.py` to return a raw RGB crop (not pre-normalized) — each detector applies its own model-specific preprocessing via the HF processor now.
- Removed the LSTM temporal-sequence buffering logic from `VideoDetectorInference` (`SEQ_LENGTH`/`frame_buffer`/`"BUFFERING (n/5)"` label) since ViT classifies one frame at a time — it doesn't apply anymore. **This is a real capability gap** — see §7 item 1, this is exactly the kind of temporal signal the roadmap wants back, done properly this time.
- `requirements.txt` was missing `fastapi`, `uvicorn`, `python-multipart` (needed for FastAPI `UploadFile`), `mss`, `customtkinter`, `winotify`, `websockets`, `transformers` — all added.
- `test_imports.py` imported deleted class names (`VideoDeepfakeDetector`, `AudioDeepfakeModel`) — rewritten to match the current API.
- `test_models.py` tested a different, unrelated audio model ID than what's actually deployed — rewritten to load from `weights/` via the real detector classes.
- `app.py`'s Gradio "About" tab and `static/index.html`'s status page had stale copy claiming the models were "Untrained" / "EfficientNet-B0" — updated to reflect ViT/Wav2Vec2 and show live `.ready` status.
- `config.py` had dead variables `VIDEO_MODEL_WEIGHTS`/`AUDIO_MODEL_WEIGHTS` pointing at a *third*, never-used naming scheme — removed with an explanatory comment.

**Verified working** (2026-09-17, on her machine, CPU): `test_imports.py` and `test_models.py` both load real weights cleanly (`[VideoDetector] Labels: {0: 'Real', 1: 'Fake'} -> fake index = 1`, `[AudioDetector] Labels: {0: 'bonafide', 1: 'spoof'} -> fake index = 1`), no `UNKNOWN FORMAT`, no random-init warnings. `server.py` starts clean and reaches `[Server] Pipelines ready.`. Deterministic (not random) output confirmed on repeated calls.

**One thing to know**: `server.py`'s `__main__` runs `uvicorn.run("server:app", reload=True)`, which spawns a reloader + worker process — you'll see every startup log line (including model loading) print twice, and both ~700MB checkpoints load into RAM twice. Harmless, just slower startup than necessary. Fine to leave for dev; consider `reload=False` for anything demo/submission-facing.

---

## 4. Directory map

```
deepfake_detector/
├── app.py                    # Gradio UI entrypoint
├── server.py                 # FastAPI + WebSocket entrypoint (this is the "real" web UI)
├── call_monitor.py           # Windows desktop app (screen+system-audio capture, toast alerts)
├── alert_system.py           # Rolling-window consensus + Windows toast notifications
├── config.py                 # All tunables: thresholds, device, frame sizes, sample rates
├── requirements.txt
├── test_imports.py           # Smoke test: do all imports + both detectors load & predict?
├── test_models.py            # Smoke test: load weights/, run one prediction each
├── test_ws.py                 # Manual WebSocket smoke test against a running server.py
├── models/
│   ├── face_extractor.py     # MTCNN face detection → raw RGB crops
│   ├── video_detector.py     # ViT-based FAKE/REAL classifier (per-frame)
│   ├── audio_detector.py     # Wav2Vec2-based FAKE/REAL classifier (per-window)
│   ├── custom_audio_net.py   # UNUSED — from-scratch CNN, kept for future retraining (see §3)
│   └── custom_video_net.py   # UNUSED — from-scratch CNN+LSTM, kept for future retraining
├── pipeline/
│   ├── video_pipeline.py     # Webcam/file → face_extractor → video_detector, draws overlays
│   └── audio_pipeline.py     # Mic/file → sliding windows → audio_detector
├── capture/
│   ├── screen_capture.py     # mss-based screen region capture (for call_monitor.py)
│   └── audio_capture.py      # WASAPI loopback system-audio capture (for call_monitor.py)
├── static/                   # server.py's frontend: index.html, script.js, styles.css
├── training/                 # Standalone Colab/Kaggle scripts — NOT run as part of the app
│   ├── train_kaggle.py, train_audio_kaggle.py    # download+fine-tune the HF checkpoints actually in use
│   ├── train_video_colab.py, train_audio_colab.py # Colab equivalents
│   ├── train_custom_video.py, train_custom_audio.py # from-scratch training for the UNUSED custom nets
│   └── README.md
└── weights/
    ├── audio_model/   # ACTIVE — Wav2Vec2ForSequenceClassification checkpoint
    └── video_model/   # ACTIVE — ViTForImageClassification checkpoint
```

---

## 5. Config knobs (`config.py`)

| Variable | Value | Meaning |
|---|---|---|
| `FAKE_THRESHOLD` | 0.5 | Video: `fake_probability` above this → label `FAKE` |
| `AUDIO_FAKE_THRESHOLD` | 0.5 | Audio: same, for `fake_probability` |
| `FRAME_SKIP` | 3 | Video pipeline only runs detection every Nth frame (perf) |
| `FACE_DETECTION_THRESHOLD` | 0.9 | MTCNN confidence floor to accept a detected face |
| `AUDIO_WINDOW_SECONDS` / `AUDIO_STRIDE_SECONDS` | 3 / 1 | Sliding window size/hop for audio |
| `SAMPLE_RATE` | 16000 | Audio pipeline internal rate |
| `VIDEO_ONNX_PATH` / `AUDIO_ONNX_PATH` | — | Dead/aspirational, no ONNX export exists yet |

---

## 6. Roadmap — what to build next, in priority order

Context for why this roadmap exists: modern deepfakes (diffusion-based face swap, few-second voice cloning) are good enough that a single end-to-end classifier trained on one generation method generalizes poorly to newer/unseen generators. The fix isn't a bigger model — it's **fusing multiple cheap, independent "tells"** so no single generator can fool every signal at once. Below, in the order to build them:

### 1. Audio-visual lip-sync consistency check — build this first
**Why first**: it's the most natural extension of what already exists — `video_pipeline.py` already has face landmarks available via MTCNN, and `audio_pipeline.py` already runs in parallel. No new capture/IO needed, just a correlation module between two streams that already exist.

**What it does**: mismatch between mouth movement and speech audio is one of the most reliable deepfake tells, especially for real-time face-swap + voice-clone combined attacks (exactly `call_monitor.py`'s threat model — a video call where both video and audio are faked).

**Concrete plan**:
- Add mouth/lip landmark extraction. `facenet_pytorch`'s `MTCNN` (already a dependency) can return landmarks (`mtcnn.detect(img, landmarks=True)` gives 5-point landmarks including mouth corners) — for finer mouth shape you may want to add `mediapipe` (lightweight, CPU-friendly, gives full lip contour) as a new dependency, or dlib's 68-point predictor. Recommend starting with MTCNN's cheap landmarks before reaching for a heavier dependency.
- Compute a per-frame "mouth openness"/motion signal (e.g. vertical lip distance, or optical flow in the mouth region) at the video frame rate.
- Compute a matching audio energy/voice-activity signal (RMS envelope or a proper VAD) at the same time resolution.
- Cross-correlate the two signals over a short rolling window (e.g. 1-2 seconds) — a real speaker has tight correlation; unsynced dubbing/lip-sync artifacts show up as lag or decorrelation.
- New file suggestion: `models/lipsync_detector.py` with a `LipSyncConsistency` class exposing `.update(face_crop_or_landmarks, audio_chunk) -> {"sync_score": float}`, wired into `alert_system.py` as a third signal alongside video/audio, and exposed in `server.py`'s `/ws/video` or a new combined endpoint.
- Test the same way as everything else: a standalone `test_lipsync.py` that feeds a known-good (real, synced) sample and a known-bad (dubbed/out-of-sync) sample and checks the score direction is right — mirror the pattern in `test_models.py`.

### 2. Frequency/residual-domain forensic cues fused with the ViT video classifier
Cheap, proven (+3.8–4.4% AUC in recent literature) generalization boost without retraining the backbone:
- Before feeding a face crop to `VideoDetectorInference`, also compute: a high-pass/residual map (crop minus a Gaussian-blurred version of itself — reveals diffusion/GAN upsampling artifacts), and a Local Binary Pattern (LBP) texture histogram (`skimage.feature.local_binary_pattern` — new dependency `scikit-image`).
- Train (or even just hand-tune) a tiny fusion layer that combines the ViT's `fake_probability` with a score derived from these two cheap features — start as a simple weighted average or logistic regression on top of [ViT score, residual-energy score, LBP-histogram score], not a full retrain.
- This is genuinely new signal, not a repackaging of what the ViT already sees — the residual/LBP path looks at pixel-level texture statistics the ViT's learned features may not emphasize.

### 3. Audio DSP handcrafted features ensembled with Wav2Vec2
Catches vocoder artifacts orthogonal to what the end-to-end model learned:
- Use `librosa` (already a dependency) to extract per-window: pitch/F0 (`librosa.pyin`), spectral centroid, energy, pause/silence ratio, jitter & shimmer (voice quality measures — may need `parselmouth`/Praat bindings, new dependency, or a hand-rolled implementation).
- Train a small classical ML model (logistic regression / gradient boosted trees via `scikit-learn`, already a transitive dependency via `librosa`) on these features.
- Combine with the Wav2Vec2 score via weighted ensemble (reference point: a comparable open-source student project, VoiceGuard-AI, weights CNN 50% / acoustic-ML 30% / DSP 20% — not gospel, but a reasonable starting split to tune from).
- New file suggestion: `models/audio_dsp_features.py`.

### 4. rPPG (remote photoplethysmography) heartbeat-consistency — optional/experimental
Real faces show subtle, blood-flow-driven color changes over time that many deepfakes fail to reproduce consistently. Treat as a bonus signal, not a primary one — published research (2023, "On Using rPPG Signals for DeepFake Detection: A Cautionary Note") found it doesn't generalize reliably alone. Worth a spike/experiment (e.g. `pyVHR` library) once items 1-3 are in, not before.

### 5. Combined risk score instead of independent REAL/FAKE per modality
Once 2+ of the above signals exist, replace the current "video says X, audio says Y, alert if both/either cross threshold" logic in `alert_system.py` with a single weighted ensemble risk score (0-100%) across all active signals, so the system degrades gracefully against a generator that only fools one signal.

### 6. Content provenance / watermark check
Some generators now embed invisible watermarks (e.g. SynthID) or C2PA content credentials in their output. Cheap to check for (when present) as a fast-path "definitely AI" signal, though coverage is inherently partial (not all tools watermark, and metadata can be stripped) — treat as a nice-to-have, not a dependency.

---

## 6a. Roadmap item 1 — Audio-visual lip-sync consistency check (DONE, 2026-09-17)

**Status: implemented and smoke-tested.** New/changed files:
- `models/lipsync_detector.py` (new) — `LipSyncConsistency` class: `update_video(frame, landmarks, timestamp=None)`, `update_audio(chunk, sr=None, timestamp=None)`, `get_sync_score() -> {"label": "SYNCED"|"OUT_OF_SYNC"|"INSUFFICIENT_DATA", "sync_score": float, "samples": int}`.
- `models/face_extractor.py` — added `extract_faces_with_landmarks()` alongside the existing `extract_faces()` (untouched, same return shape as before). New method returns the same dicts plus a `landmarks` key (5-point MTCNN landmarks in original-frame pixel coordinates).
- `pipeline/video_pipeline.py` / `pipeline/audio_pipeline.py` — added an opt-in `self.lipsync` attribute (default `None`). When a caller sets it to a `LipSyncConsistency` instance, `VideoPipeline.process_frame()` switches to `extract_faces_with_landmarks()` (same MTCNN cost) and feeds it; when unset, behavior is byte-for-byte what it was before this change.
- `alert_system.py` — added `update_lipsync()`, `lipsync_history`, `_check_lipsync_alert()` as a third, optional signal alongside the existing `update_video()`/`update_audio()` (both untouched otherwise). `get_status_summary()`/`reset()` extended to include lipsync status.
- `server.py` — wired end-to-end: a single shared `LipSyncConsistency` instance created in `lifespan`, attached to both `video_pipeline` and `audio_pipeline`; `/ws/video` reports `sync_score`/`label` in its JSON response (new `"lipsync"` key, additive — old clients reading only `image`/`result`/`fps` are unaffected); `/ws/audio` feeds it every incoming chunk.
- `test_lipsync.py` (new) — synthetic synced vs. desynced streams, asserts score direction and labels (mirrors the `test_imports.py`/`test_models.py` pattern). Passing as of this handoff.

**Key decision — why frame-differencing instead of vertical lip-opening distance:** `facenet_pytorch`'s MTCNN only returns 5-point landmarks (eyes, nose, mouth corners) — there's no top/bottom lip point, so there's no cheap *geometric* mouth-openness measurement without adding a heavier model (mediapipe/dlib), which the original roadmap note said to avoid for a first pass. Went with the roadmap's own listed alternative instead: frame-differenced grayscale energy inside the mouth bounding box (an "optical flow in the mouth region"-style motion proxy) — it captures opening/closing motion as a texture-change signal and costs nothing extra, since MTCNN already computes the 5-point landmarks as part of its normal forward pass (no second model, no second detection call). If mouth-openness precision ever becomes the bottleneck (e.g. false positives from head motion/lighting changes bleeding into the frame-diff signal), that's the point to reconsider adding `mediapipe`'s full lip contour — not before.

**Not wired yet (left for whoever's up next, or when there's time to test them live):** `app.py` (Gradio webcam handler) and `call_monitor.py` (desktop screen/system-audio monitor) don't attach a `LipSyncConsistency` instance yet — only `server.py` does. The API surface (`pipeline.lipsync = LipSyncConsistency()`, `alert_system.update_lipsync(result)`) is designed to make that a small addition, not a redesign, whenever it's needed — this was deprioritized because those entry points needed to be exercised live (webcam/screen capture) to validate, which wasn't practical to check from here the same way `server.py`'s WebSocket wiring could be verified by import/compile + the smoke test.

---

## 6b. Roadmap item 2 — Frequency/residual-domain forensic cues (DONE, 2026-09-17)

**Status: implemented and smoke-tested.** New/changed files:
- `models/forensic_features.py` (new) — `ForensicFeatureExtractor` class: `compute(face_rgb) -> {"residual_score", "lbp_score", "residual_energy", "lbp_entropy"}` and `fuse(vit_result, face_rgb, weights=None) -> {"label", "confidence", "fake_probability", "components"}` (same output shape as `VideoDetectorInference.predict()`, plus an extra `components` key — additive, doesn't change what existing callers reading `label`/`confidence`/`fake_probability` see).
- `pipeline/video_pipeline.py` — added an opt-in `self.forensics` attribute (default `None`, same pattern as `self.lipsync` from item 1). When set to a `ForensicFeatureExtractor`, `process_frame()` fuses its cues into the ViT's result right after `self.detector.predict()`; when unset, behavior is unchanged from before this item.
- `server.py` — `video_pipeline.forensics = ForensicFeatureExtractor()` set in `lifespan`, so `/ws/video` (and `/api/analyze/video`, since both go through `VideoPipeline.process_frame`) now return the fused score.
- `test_forensic_features.py` (new) — checks both cues score a flat/over-smoothed synthetic crop as more fake-like than a naturally-textured one, and that `fuse()` blends that into the ViT score without swamping a confident prediction. Passing as of this handoff.
- `requirements.txt` — added `scikit-image>=0.22.0` (new dependency, for `skimage.feature.local_binary_pattern`; it was already present in this environment's site-packages, but wasn't declared).

**What the two cues actually are:**
- *Residual energy*: crop minus a Gaussian-blurred copy of itself, mean-abs. Real camera photos carry natural sensor noise/skin micro-texture that survives this high-pass filter; over-smoothed generator output tends not to. LOW residual → flagged more fake-like.
- *LBP texture entropy*: entropy of a Local Binary Pattern histogram. Natural texture spreads across many LBP bins; unnaturally flat/synthetic texture concentrates into a few. LOW entropy → flagged more fake-like.
- Fused via a hand-tuned weighted average (`DEFAULT_WEIGHTS = {"vit": 0.7, "residual": 0.15, "lbp": 0.15}`), per the roadmap's explicit "not a full retrain" instruction — **these weights and the residual/entropy→score mappings are hand-tuned heuristics, not fit on labeled data** (there's no labeled real/fake face dataset checked into this repo to fit against — see `training/README.md`). Revisit the mapping constants in `models/forensic_features.py` (the comments there mark the hand-tuned bits) once real fake/real samples are available to calibrate against.

**Performance note (why the LBP step downsizes to 96x96 first):** benchmarked `local_binary_pattern` at the native 224x224 crop size at 8-30ms per call on this machine (varied a lot with system load — the same code measured both ways in the same session, so treat it as "noticeably not free" rather than a precise number). Since an LBP histogram is a distribution statistic, not per-pixel detail, downsizing the grayscale crop to 96x96 before computing it cut that to <1ms with no meaningful change to the resulting entropy value, bringing the whole `fuse()` call to ~3-4ms — comfortably inside the ~5fps hot-path budget. If you add another forensic cue here, benchmark it the same way (`time.perf_counter` around a warmed-up loop, not a single call) before wiring it into `process_frame()`.

**Not wired into `app.py`/`call_monitor.py` yet** — same reasoning as item 1 (§6a): those entry points weren't exercised live from here, so wiring is left as a small, mechanical follow-up (`pipeline.forensics = ForensicFeatureExtractor()`, same pattern as `server.py`'s `lifespan`).

---

## 6c. Roadmap item 3 — Audio DSP handcrafted features ensemble (DONE, 2026-09-17)

**Status: implemented and smoke-tested.** New/changed files:
- `models/audio_dsp_features.py` (new) — `AudioDSPFeatureExtractor` class: `compute(waveform, sr=None) -> {"jitter_score", "shimmer_score", "jitter", "shimmer", "voiced_frames"}` and `fuse(wav2vec_result, waveform, sr=None, weights=None) -> {"label", "confidence", "fake_probability", "components"}` (same output shape as `AudioDetectorInference.predict()`, plus extra `components` — additive, same pattern as `models/forensic_features.py` from item 2).
- `pipeline/audio_pipeline.py` — added an opt-in `self.audio_dsp` attribute (default `None`). When set, `_process_loop()`, `process_audio_file()`, and `process_audio_file_from_array()` all fuse its cues into Wav2Vec2's result right after `self.detector.predict()`; unset, behavior is unchanged from before this item.
- `server.py` — `audio_pipeline.audio_dsp = AudioDSPFeatureExtractor()` set in `lifespan`. `/ws/audio` calls `audio_pipeline.detector.predict()` directly (bypassing `AudioPipeline`'s own buffering), so it fuses inline right after that call too — both streaming paths and `/api/analyze/audio` (via `process_audio_file`) now return the fused score.
- `test_audio_dsp_features.py` (new) — checks a perfectly steady/periodic synthetic tone scores more fake-like than the same tone with realistic frame-scale pitch/amplitude perturbation added, checks `fuse()`'s shape/blending, and checks silence doesn't crash (falls back to a neutral 0.5, doesn't fabricate a reading). Passing as of this handoff.

**Deviation from the roadmap text — read before extending this:**
- **Used `librosa.yin`, not `librosa.pyin` or Praat/parselmouth.** Benchmarked `pyin` at ~190-350ms for a 3-second/16kHz window (this repo's `AUDIO_WINDOW_SECONDS`/`SAMPLE_RATE`) on this machine — far too slow to re-run every `AUDIO_STRIDE_SECONDS` (1s). `yin` measured ~10-12ms for the same window (~30x cheaper) at the cost of not providing its own voiced/unvoiced probability — this module gates voicing itself via a relative-RMS-energy threshold (`rms > 0.3 * window_max_rms`) instead. **If you add another pitch-dependent cue here, benchmark `pyin` vs `yin` again yourself before assuming either is fast enough** — don't take this note as still true if `librosa`'s implementation changes.
- **No trained classical ML model.** The roadmap suggests fitting "a small classical ML model (logistic regression / gradient boosted trees)" on the extracted features. There's no labeled real/fake voice dataset checked into this repo to fit one against — `weights/audio_model/` was fine-tuned on ASVspoof2019 *externally* (via `training/train_audio_kaggle.py`), not on data available here (see `training/README.md`). So `fuse()` is a hand-tuned weighted average (`DEFAULT_WEIGHTS = {"wav2vec": 0.75, "dsp": 0.25}`), same honest caveat as item 2's forensic cues — the jitter/shimmer→score mapping constants in the module are explicitly marked as unfit heuristics in a comment. Revisit both the mapping and the option of an actual trained fusion model once labeled data exists.
- **Only jitter + shimmer, not the full cue list the roadmap sketches** (pitch/F0, spectral centroid, energy, pause ratio). Reasoning is in the module's docstring: raw pitch/spectral-shape features are already deeply represented inside what Wav2Vec2 (a self-supervised speech model) learned during fine-tuning, so reusing them here risks *duplicating* signal the end-to-end model already has rather than adding an orthogonal one. Jitter/shimmer (cycle-to-cycle micro-instability) are a more clearly orthogonal, literature-supported tell — vocoders tend to over-smooth exactly that. If pause-ratio or another orthogonal DSP cue is added later, keep asking that same question first: "does Wav2Vec2 already effectively see this?"

**Not wired into `app.py`/`call_monitor.py` yet** — same reasoning as items 1-2 (§6a-6b): not exercised live from here. Follow-up is mechanical: `pipeline.audio_dsp = AudioDSPFeatureExtractor()`.

---

## 6d. Roadmap item 5 — Combined risk score (DONE, 2026-09-17)

**Status: implemented and smoke-tested.** This is the one item so far that changes `alert_system.py`'s actual decision logic (items 1-3 only ever *added* optional signals alongside the old logic) — the roadmap explicitly calls for a replacement once 2+ signals exist, which items 1-3 just satisfied.

**What changed:**
- Removed the old per-modality hard-threshold checks (`_check_video_alert`/`_check_audio_alert`/`_check_lipsync_alert`, and their constants `CONSECUTIVE_FAKES_TO_ALERT`/`VIDEO_ALERT_THRESHOLD`/`AUDIO_ALERT_THRESHOLD`/`CONSECUTIVE_DESYNCS_TO_ALERT`/`LIPSYNC_ALERT_SCORE`) — each modality used to need to independently cross its own threshold N times in a row.
- Added `AlertSystem.get_risk_score() -> {"risk_score": 0-100, "level": "IDLE"|"LOW"|"MEDIUM"|"HIGH", "active_signals": {name: risk_0_to_1}}`. Each of video/audio/lipsync only becomes "active" once it has `MIN_READINGS_FOR_SIGNAL` (3) rolling readings; active signals are blended via `SIGNAL_WEIGHTS = {"video": 0.4, "audio": 0.4, "lipsync": 0.2}`, **renormalized over whichever signals are currently active** — so a single strongly-fake signal alone still alerts at full weight (graceful degradation when other signals are unavailable), not just when all three agree.
- `_check_alerts()` now fires when `risk_score >= RISK_ALERT_THRESHOLD` (50, deliberately below the old single-signal 60 threshold) instead of the old boolean OR/AND of independent checks — this is the actual behavior change the roadmap wanted: **a generator that keeps every individual signal just under its own old threshold (e.g. video=0.55 AND audio=0.55, both under the old 0.6) now still gets caught**, because the combined score crosses 50 even though neither would have crossed 60 alone. `test_alert_system.py`'s second test is exactly this scenario.
- **Public API is unchanged**: `update_video()`, `update_audio()`, `update_lipsync()`, `get_status_summary()` (now also includes a `"risk"` key — additive), `reset()`, and the `AlertSystem(on_alert_callback=...)` constructor with its `(alert_type, message, details_dict)` callback signature all still work exactly as `call_monitor.py` already calls them (verified: `call_monitor.py` still compiles and its usage — `self.alert_system.update_video(result)`, `.update_audio(result)`, `.get_status_summary()["alert_active"/"alert_message"]`, `_on_alert(alert_type, message, details)` — matches what's still there).
- `test_alert_system.py` (new) — covers: no-data → IDLE; two individually-sub-threshold signals combining into a HIGH alert (the core new behavior); one strongly-fake signal alone still alerting; all-real/in-sync staying LOW; `INSUFFICIENT_DATA` lipsync readings correctly excluded rather than counted as risk-free *or* risky; `get_status_summary()`'s existing keys still present. Passing as of this handoff.

**Bug found and fixed along the way (not part of the roadmap item, but surfaced by writing the test for it):** `AlertSystem._trigger_alert()`'s console `print()` of the alert message contains emoji (🚨/⚠️) and crashes with `UnicodeEncodeError` on a plain Windows console using the default cp1252 codepage — this was pre-existing code that no prior test ever actually exercised (nothing before item 5 called an update_*() enough times in one test to make an alert actually fire). Fixed with a `_safe_print()` helper that degrades to `?`-substitution instead of crashing when the console can't encode the message — the toast notification and `alert_message` string themselves still carry the real emoji, only the console `print()` degrades. **If you add more console output to alert paths, route it through `_safe_print()` too**, or this will resurface.

---

## 7. Ground rules for whoever builds this

- **Don't touch `models/audio_detector.py` / `models/video_detector.py`'s existing `predict()` contract** (input/output shapes) — every entry point (`server.py`, `app.py`, `call_monitor.py`, both pipelines) depends on it. Add new signals as *additional* modules that consume the same inputs, not by mutating these two files' public interface.
- **Keep real-time constraints in mind.** `server.py`'s WebSocket loop targets ~5fps video / continuous audio windows on (likely) a single consumer GPU + CPU. Anything added to the hot path (per-frame or per-window) needs to be fast — prefer the cheapest signal that works, profile before adding a second heavy model.
- **`models/custom_audio_net.py` / `models/custom_video_net.py` / `training/train_custom_*.py` are unused but intentionally kept** — if she ever wants a genuinely from-scratch-trained model (for the "we built our own architecture" narrative rather than fine-tuned transfer learning), those are the starting point. Don't delete them.
- **Every new capability should get its own small smoke test** following the `test_imports.py`/`test_models.py` pattern (import it, run it on a synthetic/known input, assert the output shape and a sane direction) — that's what caught the original bugs and it's cheap insurance.
- **This is a Windows dev machine** (`C:\Users\nandi\...`, PowerShell). New dependencies go in `requirements.txt`. Double-check anything platform-specific (the existing `winotify`/WASAPI-loopback code is Windows-only by design, that's fine and expected).

---

## 8. How to verify anything you build, quickly

```powershell
cd "C:\Users\nandi\OneDrive\Desktop\swar\deepfake_detector"
pip install -r requirements.txt
python test_imports.py      # both detectors load + one prediction each, no crashes
python test_models.py       # loads weights/, runs one real prediction each, prints the dict
python server.py            # then open http://127.0.0.1:8000 and try Live Detection / Upload tabs
```

A clean run looks like: `[VideoDetector] Labels: {0: 'Real', 1: 'Fake'} -> fake index = 1`, `[AudioDetector] Labels: {0: 'bonafide', 1: 'spoof'} -> fake index = 1`, followed by `[Server] Pipelines ready.` with no `WARNING: No trained weights found`, no `UNKNOWN FORMAT`, no `Traceback`.
