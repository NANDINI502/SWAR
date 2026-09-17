"""
=============================================================
  DEEPFAKE DETECTOR - Kaggle Training Script (T4 x2 Optimized)
=============================================================

STEPS:
  1. Open Kaggle -> New Notebook
  2. Settings (right sidebar): Accelerator = GPU T4 x2, Internet = ON
  3. Add Secret: Label = "hf", Value = your HuggingFace token
  4. Cell 1: !pip install -q transformers datasets accelerate evaluate pillow scikit-learn torchvision
  5. Cell 2: Paste everything below this docstring and run
  6. Wait ~45-90 min
  7. Download deepfake_video_model.zip from Output tab
  8. Extract into: deepfake_detector/weights/video_model/
"""

# ============================================================
# AUTH - Load HuggingFace token from Kaggle Secrets
# ============================================================
import os
# Hardcoded token since Kaggle Secrets was causing issues
os.environ["HF_TOKEN"] = "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"
print("✅ HuggingFace token loaded directly!")

# ============================================================
# IMPORTS
# ============================================================
import torch, numpy as np, gc, shutil, threading, time
from PIL import Image
from datasets import load_dataset
from transformers import (
    ViTForImageClassification, ViTImageProcessor,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# HEARTBEAT — Keeps Kaggle session alive during long training
# ============================================================
def heartbeat(interval=60):
    while True:
        time.sleep(interval)
        print(f"💓 Heartbeat: {time.strftime('%H:%M:%S')} — training still running...")

hb_thread = threading.Thread(target=heartbeat, daemon=True)
hb_thread.start()
print("💓 Heartbeat started (prints every 60s to keep session alive)")

print("=" * 50)
print("  DEEPFAKE DETECTOR - Kaggle Training")
print("=" * 50)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA devices: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

# ============================================================
# CONFIG - Adjust these as needed
# ============================================================
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-Model"
DATASET_NAME = "prithivMLmods/Deepfake-vs-Real-60K"

# Save paths (use /kaggle/working/ so files persist!)
SAVE_DIR = "/kaggle/working/deepfake_model"
OUTPUT_DIR = "/kaggle/working/training_output"

# Data — use FULL 60K dataset for best accuracy
MAX_TRAIN_SAMPLES = None       # None = use ALL training samples
MAX_VAL_SAMPLES = 5000

# Training params (optimized for T4 x2)
NUM_EPOCHS = 10          # More epochs with full dataset
BATCH_SIZE = 16          # 16 per GPU, effective = 32 with 2 GPUs
LEARNING_RATE = 2e-5     # Standard fine-tuning LR
NUM_LABELS = 2
LABEL_MAP = {0: "Real", 1: "Fake"}

print(f"\nConfig: {MAX_TRAIN_SAMPLES} train, {MAX_VAL_SAMPLES} val, batch={BATCH_SIZE}, epochs={NUM_EPOCHS}")

# ============================================================
# LOAD DATASET
# ============================================================
print("\n📥 Loading dataset...")
dataset = load_dataset(DATASET_NAME)

if "test" not in dataset and "validation" in dataset:
    from datasets import DatasetDict
    dataset = DatasetDict({"train": dataset["train"], "test": dataset["validation"]})

test_key = "test" if "test" in dataset else list(dataset.keys())[-1]

if MAX_TRAIN_SAMPLES and len(dataset["train"]) > MAX_TRAIN_SAMPLES:
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(MAX_TRAIN_SAMPLES))
if MAX_VAL_SAMPLES and len(dataset[test_key]) > MAX_VAL_SAMPLES:
    dataset[test_key] = dataset[test_key].shuffle(seed=42).select(range(MAX_VAL_SAMPLES))

print(f"  Train: {len(dataset['train'])}, Val: {len(dataset[test_key])}")
gc.collect()

# ============================================================
# PREPROCESSING
# ============================================================
print("🔧 Loading processor...")
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

train_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2, 0.2, 0.15, 0.05),
    transforms.RandomRotation(5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
])

def preprocess_train(ex):
    imgs = [train_aug(i.convert("RGB")) for i in ex["image"]]
    inputs = processor(images=imgs, return_tensors="pt")
    inputs["labels"] = ex["label"]
    return inputs

def preprocess_val(ex):
    imgs = [i.convert("RGB") for i in ex["image"]]
    inputs = processor(images=imgs, return_tensors="pt")
    inputs["labels"] = ex["label"]
    return inputs

train_ds = dataset["train"].with_transform(preprocess_train)
val_ds = dataset[test_key].with_transform(preprocess_val)
print("✅ Preprocessing ready!")

# ============================================================
# LOAD MODEL
# ============================================================
print("🧠 Loading ViT model...")
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=LABEL_MAP,
    label2id={v: k for k, v in LABEL_MAP.items()},
    ignore_mismatched_sizes=True,
)

total = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total:,}")

# ============================================================
# METRICS
# ============================================================
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="weighted"),
    }

# ============================================================
# TRAINING (optimized for T4 x2)
# ============================================================
print("\n" + "=" * 50)
print("🚀 Starting training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,   # 32 per GPU x 2 GPUs = 64 effective
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=25,
    fp16=True,                       # T4 Tensor Cores accelerate fp16
    dataloader_num_workers=2,        # Kaggle can handle 2 workers
    report_to="none",
    remove_unused_columns=False,
    save_total_limit=2,
    dataloader_pin_memory=True,      # Faster GPU transfers
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

train_result = trainer.train()
print("=" * 50)

# ============================================================
# EVALUATE
# ============================================================
print("\n📊 Evaluating...")
results = trainer.evaluate()
print(f"  ✅ Accuracy: {results['eval_accuracy']:.4f} ({results['eval_accuracy']*100:.1f}%)")
print(f"  ✅ F1 Score: {results['eval_f1']:.4f}")
print(f"  📉 Loss: {results['eval_loss']:.4f}")

# ============================================================
# SAVE MODEL (to /kaggle/working/ so it persists!)
# ============================================================
print(f"\n💾 Saving model to {SAVE_DIR}...")
trainer.save_model(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

# Verify files were saved
saved_files = os.listdir(SAVE_DIR)
print(f"  Saved files: {saved_files}")

# Create zip for download
shutil.make_archive("/kaggle/working/deepfake_video_model", "zip", SAVE_DIR)
size = os.path.getsize("/kaggle/working/deepfake_video_model.zip") / (1024**2)

print("\n" + "=" * 50)
print(f"🎉 DONE! Download: deepfake_video_model.zip ({size:.1f} MB)")
print("=" * 50)
print("\nNext steps:")
print("  1. Download 'deepfake_video_model.zip' from the Output tab (right sidebar)")
print("  2. Extract into: deepfake_detector/weights/video_model/")
print("  3. Run: python call_monitor.py")
