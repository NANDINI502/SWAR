"""
Configuration for Real-Time Deepfake Detection System.
Auto-detects hardware and sets optimal parameters.
"""

import torch

# ─── Device Configuration ───────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"  # Use half-precision on GPU for speed

# ─── Video Detection Settings ───────────────────────────────────────────────────
VIDEO_FRAME_WIDTH = 640
VIDEO_FRAME_HEIGHT = 480
FACE_INPUT_SIZE = 224           # EfficientNet input size
FRAME_SKIP = 3                  # Process every Nth frame (higher = faster)
FACE_DETECTION_THRESHOLD = 0.9  # MTCNN confidence threshold
FAKE_THRESHOLD = 0.5            # Above this = classified as FAKE

# ─── Audio Detection Settings ───────────────────────────────────────────────────
SAMPLE_RATE = 16000             # 16kHz standard for speech
AUDIO_WINDOW_SECONDS = 3       # Length of audio chunks to analyze
AUDIO_STRIDE_SECONDS = 1       # Overlap stride between chunks
N_MELS = 128                   # Mel-spectrogram frequency bins
HOP_LENGTH = 512               # STFT hop length
N_FFT = 2048                   # FFT window size
AUDIO_FAKE_THRESHOLD = 0.5     # Above this = classified as FAKE

# ─── Model Paths ────────────────────────────────────────────────────────────────
# NOTE: models/video_detector.py and models/audio_detector.py load their
# HuggingFace checkpoints straight from weights/video_model/ and
# weights/audio_model/ (see their WEIGHTS_DIR class attributes) — these two
# variables used to point at unrelated .pth filenames that no script here
# ever produced, so nothing should reference them again. Kept only as
# ONNX export targets if that gets added later.
VIDEO_ONNX_PATH = "weights/video_deepfake_detector.onnx"
AUDIO_ONNX_PATH = "weights/audio_deepfake_detector.onnx"

# ─── UI Settings ────────────────────────────────────────────────────────────────
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = False

# ─── Visualization ──────────────────────────────────────────────────────────────
REAL_COLOR = (0, 255, 0)       # Green
FAKE_COLOR = (0, 0, 255)       # Red (BGR for OpenCV)
FONT_SCALE = 0.8
BOX_THICKNESS = 2
