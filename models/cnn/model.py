import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, n_features: int = 18, n_classes: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout1d(0.1),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.2),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        f   = self.encoder(x)
        avg = self.avg_pool(f).squeeze(-1)
        mx  = self.max_pool(f).squeeze(-1)
        return self.classifier(torch.cat([avg, mx], dim=1))