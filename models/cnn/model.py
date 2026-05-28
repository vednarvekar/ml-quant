import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Input shape: (Batch, 1, 60, 6)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,2), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.bn2 = nn. BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1,1)) # Reduces to (Batch, 64, 1, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x.view(x.size(0), -1) # Flatten to (Batch, 64)
    
class MultiTimeframeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Define the "Eyes"
        self.m1_eye = CNNBranch()
        self.m5_eye = CNNBranch()
        self.mH_eye = CNNBranch()

        # 2. Define the "Brain" layers (MISSING IN YOUR CODE)
        # 64 features from each of the 3 branches = 192
        self.fc1 = nn.Linear(192, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 3) # 0: Neutral, 1: Buy, 2: Sell

    def forward(self, x1, x5, xh):
        # Pass through branches
        f1 = self.m1_eye(x1)
        f5 = self.m5_eye(x5)
        fh = self.mH_eye(xh)

        # Concatenate features
        combined = torch.cat((f1, f5, fh), dim=1)
        
        # Pass through the "Brain"
        x = F.leaky_relu(self.fc1(combined))
        x = self.dropout(x)
        return self.fc2(x)
