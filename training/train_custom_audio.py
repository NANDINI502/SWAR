"""
Train Custom Audio CNN from Scratch

This script loads a deepfake audio dataset, converts waveforms to
Mel-Spectrograms on the fly with augmentations, and trains the
custom AudioDeepfakeCNN using PyTorch.

Usage:
  python training/train_custom_audio.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.custom_audio_net import AudioDeepfakeCNN

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Use a lightweight dataset for testing on local machines (or full deepfake MS-Celeb)
DATASET_NAME = "motheecreator/Deepfake-audio-dataset"
INPUT_SR = 16000
MAX_AUDIO_LEN = INPUT_SR * 3  # 3 seconds of audio

# Spectrogram Config
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# DATASET & AUGMENTATION
# ==============================================================================
class DeepfakeAudioDataset(Dataset):
    def __init__(self, hf_dataset, augment=False):
        self.dataset = hf_dataset
        self.augment = augment
        
    def __len__(self):
        return len(self.dataset)
        
    def _audio_augment(self, waveform):
        if not self.augment:
            return waveform
            
        # 1. Random gain
        if np.random.rand() < 0.5:
            gain = np.random.uniform(0.5, 1.5)
            waveform = waveform * gain
            
        # 2. Add White Noise
        if np.random.rand() < 0.3:
            noise = np.random.randn(len(waveform)) * 0.005
            waveform = waveform + noise
            
        return waveform

    def _wav_to_mel(self, waveform):
        # Create Mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=waveform, 
            sr=INPUT_SR, 
            n_mels=N_MELS, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH
        )
        
        # Convert to log scale (dB)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize to [-1, 1]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        mel_db = (mel_db * 2) - 1
        
        # Expected model input is (1, H, W). H = N_MELS.
        # W = Time steps (depends on audio length and hop_length)
        # Pad or truncate Mel array Width to fixed size (e.g. 128 for 3 seconds)
        target_width = 128
        
        if mel_db.shape[1] < target_width:
            pad_width = target_width - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :target_width]
            
        return mel_db

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Pull waveform from HF dataset
        audio_data = item['audio']['array']
        label = item['label']
        
        audio_data = np.array(audio_data, dtype=np.float32)
        
        # Truncate / Pad raw audio to MAX_AUDIO_LEN
        if len(audio_data) > MAX_AUDIO_LEN:
            if self.augment:
                # Random crop
                start = np.random.randint(0, len(audio_data) - MAX_AUDIO_LEN)
                audio_data = audio_data[start:start+MAX_AUDIO_LEN]
            else:
                audio_data = audio_data[:MAX_AUDIO_LEN]
        else:
            audio_data = np.pad(audio_data, (0, MAX_AUDIO_LEN - len(audio_data)))
            
        # Augment
        audio_data = self._audio_augment(audio_data)
        
        # To spectrogram
        mel_spec = self._wav_to_mel(audio_data)
        
        # Add channel dimension -> (1, N_MELS, TimeSteps)
        mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
        
        # Ensure label format
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return mel_tensor, label_tensor


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================
def train():
    print("=" * 60)
    print("  Custom Audio CNN Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # 1. Load Dataset
    print(f"Loading '{DATASET_NAME}'...")
    hf_dataset = load_dataset(DATASET_NAME, split='train')
    
    # Split into train/val
    hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
    train_data = hf_dataset['train']
    val_data = hf_dataset['test']
    
    print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")
    
    train_dataset = DeepfakeAudioDataset(train_data, augment=True)
    val_dataset = DeepfakeAudioDataset(val_data, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 2. Init Model
    model = AudioDeepfakeCNN(num_classes=2).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_acc = 0.0
    
    # 3. Epoch Loop
    print("\nStarting Training...")
    
    for epoch in range(EPOCHS):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct_train/total_train:.2f}%"})
            
        train_acc = 100. * correct_train / total_train
        
        # ---- EVAL ----
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  "):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = 100. * correct_val / total_val
        val_loss /= len(val_loader)
        
        print(f"\n>> Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "audio_cnn.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, save_path)
            print(f"   [!] Best model saved to {save_path} (Acc: {best_acc:.2f}%)")
            
    print(f"\nTraining Complete! Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()
