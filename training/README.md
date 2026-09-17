# 🧠 Training Deepfake Models on Google Colab

Complete step-by-step guide to train your deepfake detection models for **better real-time accuracy**.

---

## 📋 What You Need

- A Google account (for Google Colab)
- This `training/` folder with the two scripts
- ~2-5 hours of training time (free Colab T4 GPU works!)

---

## 🎥 Part 1: Train the VIDEO Deepfake Model

### Step 1 — Open Google Colab
Go to **[https://colab.research.google.com](https://colab.research.google.com)** and click **"New Notebook"**

### Step 2 — Enable GPU
1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Select **T4** (free tier) — or **A100** if you have Colab Pro
4. Click **Save**

### Step 3 — Install Dependencies
In the **first cell**, paste and run:
```python
!pip install -q transformers datasets accelerate evaluate pillow scikit-learn torchvision
```

### Step 4 — Upload the Training Script
1. Click the **📁 Files** icon on the left sidebar
2. Click the **Upload** button (⬆️ icon)
3. Upload **`train_video_colab.py`** from this folder

### Step 5 — Run the Training
In a **new cell**, paste and run:
```python
%run train_video_colab.py
```

> **⏱ Training takes ~2-4 hours** on a free T4 GPU with the full 60K dataset.
> You'll see progress logs with accuracy after each epoch.

### Step 6 — Download Your Trained Model
1. When training finishes, you'll see: `>>> Download: deepfake_video_model.zip`
2. In the **📁 Files** panel, find `deepfake_video_model.zip`
3. Right-click → **Download**

### Step 7 — Install the Model Locally
1. Extract `deepfake_video_model.zip`
2. Copy ALL the extracted files into:
   ```
   deepfake_detector/weights/video_model/
   ```
   The folder should contain files like: `config.json`, `model.safetensors`, `preprocessor_config.json`

---

## 🎤 Part 2: Train the AUDIO Deepfake Model

### Step 1 — Install Audio Dependencies
In the same Colab notebook, create a **new cell** and run:
```python
!pip install -q transformers datasets accelerate evaluate librosa soundfile scikit-learn
```

### Step 2 — Upload the Audio Training Script
Upload **`train_audio_colab.py`** to Colab Files panel

### Step 3 — Run the Training
In a **new cell**, paste and run:
```python
%run train_audio_colab.py
```

> **⏱ Training takes ~1-3 hours** on a free T4 GPU.

### Step 4 — Download & Install
1. Download `deepfake_audio_model.zip` from the Files panel
2. Extract ALL files into:
   ```
   deepfake_detector/weights/audio_model/
   ```

---

## ✅ Part 3: Verify It Works

After placing your trained models in the `weights/` folder, restart the app:

```bash
python call_monitor.py
```

You should see these lines in the output:
```
[VideoDetector] Loading model from local: weights/video_model
[AudioDetector] Loading model from local: weights/audio_model
```

If you see `Loading model from HuggingFace` instead, the weights were not placed correctly.

---

## ⚙️ Tuning Tips (Optional)

You can edit the **Configuration** section at the top of each script:

| Parameter | Default | What It Does |
|-----------|---------|---------------|
| `NUM_EPOCHS` | 8 | More epochs = better accuracy (diminishing returns after ~10) |
| `BATCH_SIZE` | 16 (video) / 8 (audio) | Reduce if you get **CUDA Out of Memory** errors |
| `LEARNING_RATE` | 1e-5 (video) / 5e-6 (audio) | Lower = more stable training |
| `MAX_TRAIN_SAMPLES` | None (full dataset) | Set a number like `5000` for a quick test run |
| `MAX_AUDIO_LENGTH` | 5 seconds | Longer = better context but more memory |

### Quick Test Run (15 minutes)
To quickly test that everything works before doing a full training run:
```python
MAX_TRAIN_SAMPLES = 1000
MAX_VAL_SAMPLES = 200
NUM_EPOCHS = 2
```

---

## 📊 Expected Results

| Model | Quick Test (1K samples) | Full Training | Full + Augmentation |
|-------|------------------------|---------------|---------------------|
| Video | ~80-85% accuracy | ~90-93% | **~94-97%** ✅ |
| Audio | ~75-80% accuracy | ~85-90% | **~90-95%** ✅ |

---

## ❓ Troubleshooting

**"CUDA out of memory"**: Reduce `BATCH_SIZE` to 8 (video) or 4 (audio)

**"RuntimeError: No GPU"**: Go to Runtime → Change runtime type → GPU

**Training is stuck at 0%**: The dataset is downloading (~2-5 GB). Wait a few minutes.

**"ModuleNotFoundError"**: Re-run the `!pip install` cell

**Colab disconnects mid-training**: Colab free tier has a ~4 hour limit. Try:
- Reducing `MAX_TRAIN_SAMPLES` to `20000`
- Using Colab Pro for longer sessions
- or keep the browser tab active (don't close it)
