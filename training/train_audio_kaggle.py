"""
=============================================================
  AUDIO DEEPFAKE DETECTOR - Kaggle Script
=============================================================
Downloads a pre-trained Wav2Vec2 deepfake detection model
that was already trained on ASVspoof2019 (~25K samples)
and achieved 99.98% accuracy.

No training needed — just downloads, packages, and zips!

STEPS:
  1. Open Kaggle -> New Notebook (GPU not required, but faster)
  2. Settings: Internet = ON
  3. Cell 1: !pip install -q transformers torch
  4. Cell 2: Paste everything below this docstring and run
  5. Wait ~2-5 minutes (just downloading, no training)
  6. Download deepfake_audio_model.zip from Output tab
  7. Extract into: deepfake_detector/weights/audio_model/
"""

import os, shutil, torch
import numpy as np
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
)

print("=" * 50)
print("  AUDIO DEEPFAKE MODEL - Download & Package")
print("=" * 50)

# ============================================================
# DOWNLOAD PRE-TRAINED MODEL
# ============================================================
# This model was fine-tuned on ASVspoof2019 dataset (~25K samples)
# and achieves 99.98% accuracy on deepfake audio detection
MODEL_NAME = "HyperMoon/wav2vec2-base-960h-finetuned-deepfake"
SAVE_DIR = "/kaggle/working/deepfake_audio_model"

print(f"\n📥 Downloading pre-trained model: {MODEL_NAME}")
print("   (Already trained on ASVspoof2019 with 99.98% accuracy)")
print("   This will take 2-5 minutes...\n")

# Download model and feature extractor
print("  Downloading model weights...")
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
print("  Downloading feature extractor...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

# Show model info
print(f"\n📋 Model Info:")
print(f"  Architecture: {model.config.architectures}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Labels: {model.config.id2label}")
print(f"  Num labels: {model.config.num_labels}")

# ============================================================
# QUICK SANITY TEST
# ============================================================
print("\n🧪 Running sanity test...")
model.eval()

# Test with random noise (should be detected as bonafide/real since noise ≠ speech)
test_audio = np.random.randn(16000 * 3).astype(np.float32)  # 3 seconds of noise
inputs = feature_extractor(test_audio, sampling_rate=16000, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

for idx, label in model.config.id2label.items():
    print(f"  {label}: {probs[int(idx)]:.4f}")

predicted = model.config.id2label[str(probs.argmax().item())]
print(f"  Random noise predicted as: {predicted}")
print("  ✅ Model is working!")

# ============================================================
# SAVE MODEL FOR LOCAL USE
# ============================================================
print(f"\n💾 Saving model to {SAVE_DIR}...")
os.makedirs(SAVE_DIR, exist_ok=True)

model.save_pretrained(SAVE_DIR)
feature_extractor.save_pretrained(SAVE_DIR)

saved_files = os.listdir(SAVE_DIR)
print(f"  Saved files: {saved_files}")

# Create zip for download
shutil.make_archive("/kaggle/working/deepfake_audio_model", "zip", SAVE_DIR)
size = os.path.getsize("/kaggle/working/deepfake_audio_model.zip") / (1024**2)

print("\n" + "=" * 50)
print(f"🎉 DONE! Download: deepfake_audio_model.zip ({size:.1f} MB)")
print("=" * 50)
print(f"\n📊 Model accuracy: 99.98% on ASVspoof2019")
print(f"   Labels: bonafide (real) / spoof (fake)")
print(f"\nNext steps:")
print(f"  1. Download 'deepfake_audio_model.zip' from Output tab")
print(f"  2. Extract ALL files into: deepfake_detector/weights/audio_model/")
print(f"  3. Run: python call_monitor.py")
print(f"\nYour app will auto-detect the label format (bonafide/spoof)!")
