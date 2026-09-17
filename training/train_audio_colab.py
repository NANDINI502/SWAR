"""
=============================================================
  AUDIO DEEPFAKE DETECTOR -- Google Colab Training Script
=============================================================
Fine-tunes a Wav2Vec2 model to detect AI-generated (deepfake)
vs real human voice for high-accuracy audio deepfake detection.

HOW TO USE IN GOOGLE COLAB:
  1. Open Google Colab: https://colab.research.google.com
  2. Go to Runtime -> Change runtime type -> GPU (T4 free tier works)
  3. Create a new notebook
  4. In the FIRST cell, paste and run:
       !pip install -q transformers datasets accelerate evaluate librosa soundfile scikit-learn
  5. In the SECOND cell, paste ALL code below this docstring and run it
  6. Wait for training to complete (~1-3 hours on T4)
  7. Download "deepfake_audio_model.zip" from the Files panel (left sidebar)
  8. Extract zip contents into your project: deepfake_detector/weights/audio_model/
  9. Restart the app -- it will auto-load your trained model!
"""

# ============================================================
# CELL 2: Imports
# ============================================================
import os
import torch
import numpy as np
import librosa
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  AUDIO DEEPFAKE DETECTOR - Training Script")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected! Training will be extremely slow.")
    print("Go to Runtime -> Change runtime type -> GPU")

# ============================================================
# CELL 3: Configuration -- ADJUST THESE FOR YOUR NEEDS
# ============================================================

# --- Model Selection ---
MODEL_NAME = "facebook/wav2vec2-base"  # Pre-trained Wav2Vec2 base

# --- Dataset ---
DATASET_NAME = "motheecreator/Deepfake-audio-dataset"  # Deepfake audio dataset

# --- Training Parameters ---
OUTPUT_DIR = "./deepfake_audio_model"
NUM_EPOCHS = 8                 # More epochs for better convergence
BATCH_SIZE = 8                 # Reduce to 4 if you get OOM errors
LEARNING_RATE = 5e-6           # Very low LR for fine-tuning Wav2Vec2
WARMUP_RATIO = 0.1             # 10% warmup
WEIGHT_DECAY = 0.01            # L2 regularization
MAX_AUDIO_LENGTH = 5           # seconds (longer clips = more context = better accuracy)
SAMPLE_RATE = 16000

# --- Data limits (set to None for full dataset) ---
MAX_TRAIN_SAMPLES = None       # None = use all training samples
MAX_VAL_SAMPLES = None         # None = use all validation samples

# --- Labels ---
LABEL_MAP = {0: "Real", 1: "Fake"}
NUM_LABELS = 2

print(f"\nConfiguration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Dataset: {DATASET_NAME}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Max audio length: {MAX_AUDIO_LENGTH}s")
print(f"  Train samples: {'ALL' if MAX_TRAIN_SAMPLES is None else MAX_TRAIN_SAMPLES}")
print(f"  Val samples: {'ALL' if MAX_VAL_SAMPLES is None else MAX_VAL_SAMPLES}")

# ============================================================
# CELL 4: Load dataset
# ============================================================
print("\n" + "=" * 60)
print("Loading dataset...")
try:
    dataset = load_dataset(DATASET_NAME)
    print(f"Splits: {list(dataset.keys())}")
    for split in dataset:
        print(f"  {split}: {len(dataset[split])} samples")
except Exception as e:
    print(f"Error loading primary dataset: {e}")
    print("Trying alternative dataset...")
    DATASET_NAME = "Hemg/deepfake-audio-dataset"
    dataset = load_dataset(DATASET_NAME)
    print(f"Loaded alternative: {list(dataset.keys())}")

# Handle different split names
if "train" not in dataset and "validation" not in dataset:
    full = list(dataset.values())[0]
    split = full.train_test_split(test_size=0.2, seed=42)
    dataset = split

# Create test split if missing
if "train" in dataset and "test" not in dataset and "validation" not in dataset:
    split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    from datasets import DatasetDict
    dataset = DatasetDict({"train": split["train"], "test": split["test"]})

test_key = "test" if "test" in dataset else "validation"

# Sample if limits are set
if MAX_TRAIN_SAMPLES and "train" in dataset and len(dataset["train"]) > MAX_TRAIN_SAMPLES:
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(MAX_TRAIN_SAMPLES))
if MAX_VAL_SAMPLES and test_key in dataset and len(dataset[test_key]) > MAX_VAL_SAMPLES:
    dataset[test_key] = dataset[test_key].shuffle(seed=42).select(range(MAX_VAL_SAMPLES))

print(f"\nFinal dataset:")
print(f"  Train: {len(dataset['train'])} samples")
print(f"  Test:  {len(dataset[test_key])} samples")

# ============================================================
# CELL 5: Load feature extractor and preprocess with augmentation
# ============================================================
print("\nLoading Wav2Vec2 feature extractor...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

# Ensure audio column is decoded at 16kHz
for split in dataset:
    if "audio" in dataset[split].column_names:
        dataset[split] = dataset[split].cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

def audio_augment(waveform, sr):
    """Apply audio augmentation for training robustness."""
    # Random gain adjustment
    if np.random.random() < 0.5:
        gain = np.random.uniform(0.8, 1.2)
        waveform = waveform * gain

    # Add light noise
    if np.random.random() < 0.3:
        noise = np.random.randn(len(waveform)) * 0.005
        waveform = waveform + noise

    # Random time shift
    if np.random.random() < 0.3:
        shift = int(sr * np.random.uniform(-0.1, 0.1))
        waveform = np.roll(waveform, shift)

    return waveform.astype(np.float32)

def preprocess(examples, augment=False):
    """Convert audio to model input format."""
    audio_arrays = []

    for i in range(len(examples.get("audio", []))):
        audio = examples["audio"][i]
        if isinstance(audio, dict):
            arr = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
        else:
            arr = np.array(audio, dtype=np.float32)
            sr = SAMPLE_RATE

        # Resample if needed
        if sr != SAMPLE_RATE:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Truncate/pad to fixed length
        max_len = SAMPLE_RATE * MAX_AUDIO_LENGTH
        if len(arr) > max_len:
            # Random crop during training, center crop during eval
            if augment:
                start = np.random.randint(0, len(arr) - max_len)
                arr = arr[start:start + max_len]
            else:
                arr = arr[:max_len]
        elif len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)))

        # Apply augmentation during training
        if augment:
            arr = audio_augment(arr, SAMPLE_RATE)

        # Normalize
        max_val = np.max(np.abs(arr))
        if max_val > 0:
            arr = arr / max_val

        audio_arrays.append(arr)

    # Get labels
    label_key = "label"
    for key in ["label", "labels", "target", "class"]:
        if key in examples:
            label_key = key
            break

    # Process through feature extractor
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        max_length=SAMPLE_RATE * MAX_AUDIO_LENGTH,
        truncation=True,
    )
    inputs["labels"] = examples[label_key]
    return inputs

def preprocess_train(examples):
    return preprocess(examples, augment=True)

def preprocess_val(examples):
    return preprocess(examples, augment=False)

print("Applying preprocessing transforms...")
train_dataset = dataset["train"].with_transform(preprocess_train)
val_dataset = dataset[test_key].with_transform(preprocess_val)
print("Done!")

# ============================================================
# CELL 6: Load model
# ============================================================
print("\nLoading Wav2Vec2 model...")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=LABEL_MAP,
    label2id={v: k for k, v in LABEL_MAP.items()},
)

# Freeze the feature extractor (only train classifier + some encoder layers)
model.freeze_feature_encoder()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

# ============================================================
# CELL 7: Define metrics
# ============================================================
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.label_ids
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# ============================================================
# CELL 8: Training
# ============================================================
print("\n" + "=" * 60)
print("Setting up training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,                   # 0 = no multiprocessing (avoids RAM issues on Colab)
    report_to="none",
    remove_unused_columns=False,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print(f"Training for {NUM_EPOCHS} epochs...")
print(f"Estimated time: ~{NUM_EPOCHS * 10}-{NUM_EPOCHS * 25} minutes on T4 GPU")
print("=" * 60)
train_result = trainer.train()
print("=" * 60)
print(f"\nTraining complete!")
print(f"Train loss: {train_result.training_loss:.4f}")

# ============================================================
# CELL 9: Evaluate with detailed report
# ============================================================
print("\n" + "=" * 60)
print("Evaluating on test set...")
eval_results = trainer.evaluate()
print(f"  Accuracy: {eval_results['eval_accuracy']:.4f} ({eval_results['eval_accuracy']*100:.1f}%)")
print(f"  F1 Score: {eval_results['eval_f1']:.4f}")
print(f"  Eval Loss: {eval_results['eval_loss']:.4f}")

# Detailed classification report
print("\nDetailed Classification Report:")
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
print(classification_report(predictions.label_ids, preds, target_names=["Real", "Fake"]))

# ============================================================
# CELL 10: Save model
# ============================================================
SAVE_PATH = "./deepfake_audio_model_final"
print(f"\nSaving model to {SAVE_PATH}...")
trainer.save_model(SAVE_PATH)
feature_extractor.save_pretrained(SAVE_PATH)
print("Model saved!")

# Create zip for download
import shutil
shutil.make_archive("deepfake_audio_model", "zip", SAVE_PATH)
file_size = os.path.getsize("deepfake_audio_model.zip") / (1024 * 1024)
print(f"\n>>> Download: deepfake_audio_model.zip ({file_size:.1f} MB)")
print(">>> Extract contents into: deepfake_detector/weights/audio_model/")

# ============================================================
# CELL 11: Test prediction
# ============================================================
print("\n" + "=" * 60)
print("Quick test with sample audio...")

# Test with random audio (sanity check)
test_audio = np.random.randn(SAMPLE_RATE * 3).astype(np.float32)
inputs = feature_extractor(test_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
if torch.cuda.is_available():
    model = model.to("cuda")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_idx = probs.argmax(-1).item()

print(f"Random noise prediction: {LABEL_MAP[pred_idx]}")
print(f"Probabilities: Real={probs[0][0]:.4f}, Fake={probs[0][1]:.4f}")

print("\n" + "=" * 60)
print("DONE! Your audio deepfake model is ready.")
print("=" * 60)
print("\nNext steps:")
print("  1. Download 'deepfake_audio_model.zip' from the Files panel")
print("  2. Extract into: deepfake_detector/weights/audio_model/")
print("  3. Restart the app: python call_monitor.py")
