"""
Spatio-Temporal Video Deepfake Model

A custom PyTorch architecture combining a CNN feature extractor
(to process individual frames) with an LSTM (to process temporal sequences).

Expects input tensors of shape (Batch, Sequence_Length, Channels, Height, Width)
Typically: (B, 10, 3, 224, 224) representing 10 consecutive frames.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class VideoDeepfakeLSTM(nn.Module):
    def __init__(self, sequence_length: int = 10, hidden_dim: int = 256, num_classes: int = 2):
        super(VideoDeepfakeLSTM, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # 1. Spatial Feature Extractor (CNN)
        # We use a lightweight pre-trained CNN (EfficientNet-B0) up to the final pooling layer.
        # This gives us a 1280-dimensional feature vector per frame.
        base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1]) # Remove classifier
        
        # Freeze CNN weights initially to focus on training the Temporal LSTM
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        cnn_out_dim = 1280
            
        # 2. Temporal Sequence Processor (LSTM)
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=2,           # 2 stacked LSTM layers
            batch_first=True,       # Input shape: (Batch, Seq, Features)
            dropout=0.3
        )
        
        # 3. Classifier Head
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (B, SeqLen, C, H, W)
        batch_size, seq_len, c, h, w = x.size()
        
        # We need to process each frame through the CNN.
        # Instead of a loop, we reshape the sequence into a large batch.
        # Reshape: (B * SeqLen, C, H, W)
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features
        cnn_features = self.feature_extractor(x_reshaped) 
        # cnn_features shape: (B * SeqLen, 1280, 1, 1)
        
        cnn_features = cnn_features.view(batch_size * seq_len, -1) # Flatten (B * SeqLen, 1280)
        
        # Reshape back to sequences: (B, SeqLen, 1280)
        sequence_features = cnn_features.view(batch_size, seq_len, -1)
        
        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(sequence_features)
        # lstm_out shape: (B, SeqLen, HiddenDim)
        
        # We only care about the hidden state at the final time step
        final_time_step_out = lstm_out[:, -1, :] # Shape: (B, HiddenDim)
        
        # Classification
        out = self.dropout(self.relu(self.fc1(final_time_step_out)))
        out = self.fc2(out)
        
        return out

if __name__ == "__main__":
    # Sanity check
    model = VideoDeepfakeLSTM(sequence_length=5)
    # Dummy tensor: Batch of 2, 5 frames, 3 RGB channels, 224x224 pixels
    dummy_input = torch.randn(2, 5, 3, 224, 224)
    output = model(dummy_input)
    print("Video LSTM Model instantiated successfully.")
    print(f"Output shape needs to be (2, 2): {output.shape}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
