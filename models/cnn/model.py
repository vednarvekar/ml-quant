import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── MODEL DESIGN: TEMPORAL-COARSE HYBRID CNN ────────────────────────────────
class CNNModel(nn.Module):
    def __init__(self, n_features: int = 12, n_classes: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            # Conv block 1
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            
            # Conv block 2 + pooling (Compresses lookback length: 60 -> 30)
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),         
            
            # Conv block 3 + pooling (Compresses lookback length: 30 -> 15)
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),         
        )
        
        # 64 channels * 15 remaining downsampled structural steps = 960 features.
        # This architecture runs incredibly fast on CPU and prevents extreme overfitting.
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(64 * 15, 64),
        #     nn.GELU(),
        #     nn.Dropout(0.4),  # Strong regularizer to block financial noise patterns6
        #     nn.Linear(64, n_classes),
        # )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 15, 512),  # wider: 960 → 128 instead of 960 → 64
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),      # extra step down: 512 → 128
            nn.GELU(),
            nn.Linear(256, 128), # final: 128 → 3
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))