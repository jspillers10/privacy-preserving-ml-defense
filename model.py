"""
Fashion-MNIST CNN Architecture

Required architecture for HTB Privacy Defense Challenge.
Must match server validation exactly.
"""

import torch
import torch.nn as nn


class FashionMNISTCNN(nn.Module):
    """
    Convolutional Neural Network for Fashion-MNIST classification.
    
    Architecture:
        Conv1: 1→32 channels, 3x3 kernel, padding=1
        Conv2: 32→64 channels, 3x3 kernel, padding=1
        Conv3: 64→64 channels, 3x3 kernel, padding=1
        Each conv followed by ReLU and 2x2 MaxPool
        FC1: 576→128 (flattened: 64*3*3)
        FC2: 128→10 (output classes)
    
    Total Parameters: ~122,858
    """
    
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Fully connected layers
        # After 3 pooling ops: 28→14→7→3
        # Feature map: 64 × 3 × 3 = 576
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, 1, 28, 28)
        
        Returns:
            Output logits (batch_size, 10)
        """
        # Conv block 1: 28×28 → 14×14
        x = self.pool(self.relu(self.conv1(x)))
        
        # Conv block 2: 14×14 → 7×7
        x = self.pool(self.relu(self.conv2(x)))
        
        # Conv block 3: 7×7 → 3×3
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten and fully connected layers
        x = x.view(-1, 64 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


if __name__ == "__main__":
    # Test architecture
    model = FashionMNISTCNN()
    test_input = torch.randn(4, 1, 28, 28)
    output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
