"""
Train Custom Video CNN+LSTM from Scratch

This script builds a dataset that loads sequences of image frames
and trains a Spatio-Temporal PyTorch model on them.

Usage:
  python training/train_custom_video.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.custom_video_net import VideoDeepfakeLSTM

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATASET_NAME = "prithivMLmods/Deepfake-vs-Real-60K"
IMG_SIZE = 224
SEQ_LENGTH = 5     # Number of frames to process in a single RNN sequence
BATCH_SIZE = 8     # Batch size of sequences
EPOCHS = 15
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# DATASET & AUGMENTATION
# ==============================================================================
class VideoSequenceDataset(Dataset):
    """
    Since the dataset contains isolated images, we simulate 'sequences'
    by combining N random images of the same class together.
    In a real video dataset (like FaceForensics++), you would load consecutive frames.
    """
    def __init__(self, hf_dataset, transform=None, seq_length=5):
        self.dataset = hf_dataset
        self.transform = transform
        self.seq_length = seq_length
        
        # Group indices by label to create consistent sequences
        self.real_indices = [i for i, item in enumerate(self.dataset) if item['label'] == 0]
        self.fake_indices = [i for i, item in enumerate(self.dataset) if item['label'] == 1]
        
    def __len__(self):
        # We define an arbitrary epoch size (e.g. 5000 sequences)
        return 5000
        
    def __getitem__(self, idx):
        # 50/50 chance of generating a REAL or FAKE sequence
        label = 0 if np.random.rand() < 0.5 else 1
        
        # Sample self.seq_length indices from the appropriate class pool
        if label == 0:
            indices = np.random.choice(self.real_indices, self.seq_length, replace=True)
        else:
            indices = np.random.choice(self.fake_indices, self.seq_length, replace=True)
            
        sequence_tensors = []
        for i in indices:
            img = self.dataset[int(i)]['image'].convert('RGB')
            if self.transform:
                img_t = self.transform(img)
            else:
                img_t = transforms.ToTensor()(img)
            sequence_tensors.append(img_t)
            
        # Stack into shape: (SEQ_LENGTH, C, H, W)
        seq_tensor = torch.stack(sequence_tensors)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return seq_tensor, label_tensor

# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================
def train():
    print("=" * 60)
    print("  Custom Video CNN+LSTM Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # 1. Loading Image Base Dataset
    print(f"Loading '{DATASET_NAME}'...")
    hf_dataset = load_dataset(DATASET_NAME, split='train')
    
    # Split
    hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
    train_data = hf_dataset['train']
    val_data = hf_dataset['test']
    
    # Define image transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = VideoSequenceDataset(train_data, transform=train_transform, seq_length=SEQ_LENGTH)
    val_dataset = VideoSequenceDataset(val_data, transform=val_transform, seq_length=SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 2. Init Model
    model = VideoDeepfakeLSTM(sequence_length=SEQ_LENGTH, hidden_dim=256, num_classes=2).to(DEVICE)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (LSTM Head): {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_acc = 0.0
    
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
            save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "video_lstm.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, save_path)
            print(f"   [!] Best model saved to {save_path} (Acc: {best_acc:.2f}%)")

if __name__ == "__main__":
    train()
