"""
Video Deepfake Detector - ViT (HuggingFace)
============================================
Loads the fine-tuned ViTForImageClassification checkpoint that was
downloaded/trained via training/train_kaggle.py (or train_video_colab.py)
and saved to weights/video_model/. That checkpoint is what actually exists
on disk — it classifies a single face crop per call, not a frame sequence.

(The earlier "custom CNN+LSTM" path expected weights/video_lstm.pth, which
no script here produces, so it silently ran on a randomly initialized
network. It was also being fed an already ImageNet-normalized torch.Tensor
from FaceExtractor while only accepting a raw ndarray/PIL.Image — so every
call actually short-circuited to "UNKNOWN FORMAT" before it even reached
the model. Both bugs are fixed by this rewrite + the matching change in
face_extractor.py.)
"""

import os
import torch
import numpy as np
from PIL import Image

from config import DEVICE, FAKE_THRESHOLD

try:
    from transformers import ViTForImageClassification, ViTImageProcessor
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class VideoDetectorInference:
    """
    Deepfake face detector using the locally fine-tuned ViT checkpoint in
    weights/video_model/ (id2label: {0: "Real", 1: "Fake"}).
    Classifies each face crop independently (no temporal buffering).
    """

    WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "video_model")

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.ready = False
        self.fake_index = 1  # overwritten below once id2label is read

        print("[VideoDetector] Initializing ViT deepfake classifier...")
        print(f"[VideoDetector] Device: {device}")

        if not _HAS_TRANSFORMERS:
            print("[VideoDetector] ERROR: `transformers` is not installed. "
                  "Run `pip install transformers` (see requirements.txt).")
            return

        if not os.path.isdir(self.WEIGHTS_DIR) or not os.listdir(self.WEIGHTS_DIR):
            print(f"[VideoDetector] ERROR: No trained weights found at {self.WEIGHTS_DIR}. "
                  f"Run training/train_kaggle.py (or train_video_colab.py) and extract "
                  f"the result into that folder.")
            return

        try:
            print(f"[VideoDetector] Loading trained weights from {self.WEIGHTS_DIR}")
            self.processor = ViTImageProcessor.from_pretrained(self.WEIGHTS_DIR)
            self.model = ViTForImageClassification.from_pretrained(self.WEIGHTS_DIR)

            id2label = {int(k): v for k, v in self.model.config.id2label.items()}
            fake_labels = {"fake", "spoof", "ai", "synthetic"}
            match = [idx for idx, label in id2label.items() if label.strip().lower() in fake_labels]
            self.fake_index = match[0] if match else 1
            print(f"[VideoDetector] Labels: {id2label} -> fake index = {self.fake_index}")

            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            else:
                self.model = self.model.to("cpu")
                self.device = "cpu"

            self.model.eval()
            self.ready = True

        except Exception as e:
            print(f"[VideoDetector] Error initializing model: {e}")
            self.ready = False

    @torch.no_grad()
    def predict(self, face_image) -> dict:
        """
        Predict if a single face crop is real or fake.

        Args:
            face_image: numpy array (H, W, C) RGB uint8, or a PIL Image.
                        (FaceExtractor.extract_faces() returns a raw RGB
                        crop in this format — do not pre-normalize it.)

        Returns:
            dict with 'label', 'confidence', 'fake_probability'
        """
        if not self.ready:
            return {"label": "ERROR", "confidence": 0.0, "fake_probability": 0.5}

        try:
            if isinstance(face_image, np.ndarray):
                pil_image = Image.fromarray(face_image.astype(np.uint8)).convert("RGB")
            elif isinstance(face_image, Image.Image):
                pil_image = face_image.convert("RGB")
            elif isinstance(face_image, torch.Tensor):
                # Defensive fallback in case an already-normalized tensor is
                # ever passed in again — undo a [-1,1]/ImageNet-ish scale and
                # convert back to a uint8 image before handing it to the
                # HF processor (which expects raw pixels, not pre-normalized).
                arr = face_image.detach().cpu()
                if arr.dim() == 3 and arr.shape[0] in (1, 3):
                    arr = arr.permute(1, 2, 0)
                arr = arr.numpy()
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
                pil_image = Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")
            else:
                return {"label": "UNKNOWN FORMAT", "confidence": 0.0, "fake_probability": 0.5}

            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

            fake_prob = probs[self.fake_index].item()
            is_fake = fake_prob > FAKE_THRESHOLD
            label = "FAKE" if is_fake else "REAL"
            confidence = fake_prob if is_fake else (1.0 - fake_prob)

            return {
                "label": label,
                "confidence": float(confidence),
                "fake_probability": float(fake_prob),
            }

        except Exception as e:
            print(f"[VideoDetector] Prediction error: {e}")
            return {"label": "PRED ERROR", "confidence": 0.0, "fake_probability": 0.5}
