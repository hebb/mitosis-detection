import torch
from torch import nn


class MitosisCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
