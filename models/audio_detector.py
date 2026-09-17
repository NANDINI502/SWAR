"""
Audio Deepfake Detector - Wav2Vec2 (HuggingFace)
=================================================
Loads the fine-tuned Wav2Vec2ForSequenceClassification checkpoint that was
downloaded/trained via training/train_audio_kaggle.py (or train_audio_colab.py)
and saved to weights/audio_model/. That checkpoint is what actually exists on
disk — the earlier "from-scratch CNN" path looked for weights/audio_cnn.pth,
which is never produced by any script here, so it silently ran on a randomly
initialized network (hence the hallucinated output).
"""

import os
import torch
import numpy as np
import librosa

from config import DEVICE, AUDIO_FAKE_THRESHOLD, SAMPLE_RATE

try:
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class AudioDetectorInference:
    """
    Deepfake voice detector using the locally fine-tuned Wav2Vec2 checkpoint
    in weights/audio_model/ (id2label: {0: "bonafide", 1: "spoof"}).
    """

    WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "audio_model")

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.ready = False
        self.fake_index = 1  # overwritten below once id2label is read

        print("[AudioDetector] Initializing Wav2Vec2 deepfake classifier...")
        print(f"[AudioDetector] Device: {device}")

        if not _HAS_TRANSFORMERS:
            print("[AudioDetector] ERROR: `transformers` is not installed. "
                  "Run `pip install transformers` (see requirements.txt).")
            return

        if not os.path.isdir(self.WEIGHTS_DIR) or not os.listdir(self.WEIGHTS_DIR):
            print(f"[AudioDetector] ERROR: No trained weights found at {self.WEIGHTS_DIR}. "
                  f"Run training/train_audio_kaggle.py (or train_audio_colab.py) and extract "
                  f"the result into that folder.")
            return

        try:
            print(f"[AudioDetector] Loading trained weights from {self.WEIGHTS_DIR}")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.WEIGHTS_DIR)
            self.model = AutoModelForAudioClassification.from_pretrained(self.WEIGHTS_DIR)

            # Figure out which output index corresponds to "fake" instead of assuming
            # index 1 — id2label for this checkpoint is {0: "bonafide", 1: "spoof"}.
            id2label = {int(k): v for k, v in self.model.config.id2label.items()}
            fake_labels = {"spoof", "fake", "ai", "synthetic", "ai voice", "aivoice"}
            match = [idx for idx, label in id2label.items() if label.strip().lower() in fake_labels]
            self.fake_index = match[0] if match else 1
            print(f"[AudioDetector] Labels: {id2label} -> fake index = {self.fake_index}")

            self.target_sr = self.feature_extractor.sampling_rate

            if device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            else:
                self.model = self.model.to("cpu")
                self.device = "cpu"

            self.model.eval()
            self.ready = True

        except Exception as e:
            print(f"[AudioDetector] Error loading model: {e}")
            self.ready = False

    @torch.no_grad()
    def predict(self, audio_waveform: np.ndarray, sr: int = None) -> dict:
        """
        Predict if audio is real or fake.

        Args:
            audio_waveform: 1D numpy array of audio samples
            sr: Sample rate of input audio (will resample to the model's rate if needed)

        Returns:
            dict with 'label', 'confidence', 'fake_probability'
        """
        if not self.ready:
            return {"label": "ERROR", "confidence": 0.0, "fake_probability": 0.5}

        try:
            if sr is None:
                sr = SAMPLE_RATE

            waveform = audio_waveform.astype(np.float32)

            if sr != self.target_sr:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sr)

            inputs = self.feature_extractor(
                waveform,
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

            fake_prob = probs[self.fake_index].item()
            is_fake = fake_prob > AUDIO_FAKE_THRESHOLD
            label = "FAKE" if is_fake else "REAL"
            confidence = fake_prob if is_fake else (1.0 - fake_prob)

            return {
                "label": label,
                "confidence": float(confidence),
                "fake_probability": float(fake_prob),
            }

        except Exception as e:
            print(f"[AudioDetector] Prediction error: {e}")
            return {"label": "PRED ERROR", "confidence": 0.0, "fake_probability": 0.5}
