"""
=============================================================
  VIDEO DEEPFAKE DETECTOR -- Google Colab Training Script
=============================================================
Fine-tunes a ViT (Vision Transformer) model on real vs fake
face images for high-accuracy deepfake detection.

HOW TO USE IN GOOGLE COLAB:
  1. Open Google Colab: https://colab.research.google.com
  2. Go to Runtime -> Change runtime type -> GPU (T4 free tier works)
  3. Create a new notebook
  4. In the FIRST cell, paste and run:
       !pip install -q transformers datasets accelerate evaluate pillow scikit-learn torchvision
  5. In the SECOND cell, paste ALL code below this docstring and run it
  6. Wait for training to complete (~2-4 hours on T4 for full dataset)
  7. Download "deepfake_video_model.zip" from the Files panel (left sidebar)
  8. Extract zip contents into your project: deepfake_detector/weights/video_model/
  9. Restart the app -- it will auto-load your trained model!
"""

# ============================================================
# CELL 2: HuggingFace Authentication (for gated datasets)
# ============================================================
import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("hf")
    print("HuggingFace token loaded from Kaggle secrets!")
except Exception:
    print("Not running on Kaggle or secret 'hf' not found.")
    print("Set HF_TOKEN manually if using a gated dataset.")

# ============================================================
# CELL 3: Imports
# ============================================================
import gc
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  VIDEO DEEPFAKE DETECTOR - Training Script")
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
# Option 1 (Recommended): Start from a model already fine-tuned on deepfakes
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-Model"

# Option 2: Start from scratch with base ViT (slower convergence)
# MODEL_NAME = "google/vit-base-patch16-224-in21k"

# Option 3: Larger model for higher accuracy (needs >12GB VRAM, use Colab Pro)
# MODEL_NAME = "google/vit-large-patch16-224-in21k"

# --- Dataset ---
DATASET_NAME = "prithivMLmods/Deepfake-vs-Real-60K"  # 60K real/fake face images

# --- Training Parameters ---
OUTPUT_DIR = "./deepfake_video_model"
NUM_EPOCHS = 8                 # More epochs for better convergence
BATCH_SIZE = 8                 # Keep at 8 for Colab free tier (12GB RAM)
LEARNING_RATE = 1e-5           # Lower LR for fine-tuning pre-trained models
WARMUP_RATIO = 0.1             # 10% of training steps for warmup
WEIGHT_DECAY = 0.01            # L2 regularization

# --- Data limits ---
# Colab free tier has ~12GB RAM. Full 60K dataset will crash it.
# 15K samples gives great accuracy while fitting in memory.
# If you have Colab Pro (50GB+ RAM), set both to None for full dataset.
MAX_TRAIN_SAMPLES = 15000      # 15K training samples (fits Kaggle GPU)
MAX_VAL_SAMPLES = 3000         # 3K validation samples

# --- Labels ---
LABEL_MAP = {0: "Real", 1: "Fake"}
NUM_LABELS = 2

print(f"\nConfiguration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Dataset: {DATASET_NAME}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Train samples: {'ALL' if MAX_TRAIN_SAMPLES is None else MAX_TRAIN_SAMPLES}")
print(f"  Val samples: {'ALL' if MAX_VAL_SAMPLES is None else MAX_VAL_SAMPLES}")

# ============================================================
# CELL 4: Load dataset
# ============================================================
print("\n" + "=" * 60)
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

for split_name in dataset:
    print(f"  {split_name}: {len(dataset[split_name])} samples")

# Ensure we have train and test splits
if "train" not in dataset:
    from datasets import concatenate_datasets
    all_data = concatenate_datasets([dataset[s] for s in dataset])
    dataset = all_data.train_test_split(test_size=0.2, seed=42)

if "test" not in dataset and "validation" in dataset:
    from datasets import DatasetDict
    dataset = DatasetDict({"train": dataset["train"], "test": dataset["validation"]})

test_key = "test" if "test" in dataset else list(dataset.keys())[-1]

# Sample if limits are set
if MAX_TRAIN_SAMPLES and len(dataset["train"]) > MAX_TRAIN_SAMPLES:
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(MAX_TRAIN_SAMPLES))
if MAX_VAL_SAMPLES and len(dataset[test_key]) > MAX_VAL_SAMPLES:
    dataset[test_key] = dataset[test_key].shuffle(seed=42).select(range(MAX_VAL_SAMPLES))

print(f"\nFinal dataset:")
print(f"  Train: {len(dataset['train'])} samples")
print(f"  Test:  {len(dataset[test_key])} samples")

# Free unused memory
gc.collect()

# ============================================================
# CELL 5: Load processor and define preprocessing with augmentation
# ============================================================
print("\nLoading ViT processor...")
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

# Data augmentation for training (improves robustness to video compression, lighting, etc.)
train_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.RandomGrayscale(p=0.05),
])

def preprocess_train(examples):
    """Preprocess with data augmentation for training."""
    images = examples["image"]
    images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
    # Apply augmentation
    images = [train_augment(img) for img in images]
    inputs = processor(images=images, return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs

def preprocess_val(examples):
    """Preprocess without augmentation for validation."""
    images = examples["image"]
    images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
    inputs = processor(images=images, return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs

print("Applying preprocessing transforms...")
train_dataset = dataset["train"].with_transform(preprocess_train)
val_dataset = dataset[test_key].with_transform(preprocess_val)
print("Done!")

# ============================================================
# CELL 6: Load model
# ============================================================
print("\nLoading ViT model...")
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=LABEL_MAP,
    label2id={v: k for k, v in LABEL_MAP.items()},
    ignore_mismatched_sizes=True,
)

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
    per_device_eval_batch_size=BATCH_SIZE * 2,    # Larger batch for eval (no gradients)
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",                   # Cosine annealing for smoother convergence
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
    save_total_limit=2,                           # Keep only best 2 checkpoints to save disk
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Stop if no improvement for 3 epochs
)

print(f"Training for {NUM_EPOCHS} epochs...")
print(f"Estimated time: ~{NUM_EPOCHS * 20}-{NUM_EPOCHS * 40} minutes on T4 GPU")
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
SAVE_PATH = "./deepfake_video_model_final"
print(f"\nSaving model to {SAVE_PATH}...")
trainer.save_model(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)
print("Model saved!")

# Create zip for easy download
import shutil
shutil.make_archive("deepfake_video_model", "zip", SAVE_PATH)
file_size = os.path.getsize("deepfake_video_model.zip") / (1024 * 1024)
print(f"\n>>> Download: deepfake_video_model.zip ({file_size:.1f} MB)")
print(">>> Extract contents into: deepfake_detector/weights/video_model/")

# ============================================================
# CELL 11: Test prediction
# ============================================================
print("\n" + "=" * 60)
print("Quick test with sample images...")
from transformers import pipeline as hf_pipeline

pipe = hf_pipeline(
    "image-classification",
    model=SAVE_PATH,
    device=0 if torch.cuda.is_available() else -1,
)

# Test with a blank image (sanity check)
test_img = Image.new("RGB", (224, 224), "white")
result = pipe(test_img)
print(f"Blank image prediction: {result}")

# Test with a sample from the validation set if available
try:
    test_sample = dataset[test_key][0]
    test_image = test_sample["image"].convert("RGB")
    result = pipe(test_image)
    true_label = LABEL_MAP[test_sample["label"]]
    print(f"Sample prediction: {result} (true label: {true_label})")
except Exception:
    pass

print("\n" + "=" * 60)
print("DONE! Your video deepfake model is ready.")
print("=" * 60)
print("\nNext steps:")
print("  1. Download 'deepfake_video_model.zip' from the Files panel")
print("  2. Extract into: deepfake_detector/weights/video_model/")
print("  3. Restart the app: python call_monitor.py")
