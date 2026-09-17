"""
Audio Deepfake CNN

A custom PyTorch Convolutional Neural Network built from scratch.
Expects input tensors of shape (Batch, 1, Mel_Bins, Time_Steps).
Typically: (B, 1, 128, 128) representing a 3-second audio chunk.
"""

import torch
import torch.nn as nn

class AudioDeepfakeCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(AudioDeepfakeCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.AdaptiveAvgPool2d((4, 4)) # Forces output to (B, 256, 4, 4)
        
        # Classifier Head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # x shape: (B, 1, H, W) -> e.g., (B, 1, 128, 128)
        
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        
        x = self.flatten(x)
        x = self.dropout(self.relu_fc1(self.fc1(x)))
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # Sanity check
    model = AudioDeepfakeCNN()
    # Dummy tensor: Batch of 4, 1 channel, 128 mel bins, 128 time frames
    dummy_input = torch.randn(4, 1, 128, 128)
    output = model(dummy_input)
    print(f"Model instantiated successfully.")
    print(f"Output shape needs to be (4, 2): {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
