import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_layers: int, base_channels: int, num_classes: int):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = 3
        current_channels = base_channels

        for i in range(num_layers):
            self.layers.append(
                nn.Conv2d(
                    in_channels, out_channels=current_channels, kernel_size=3, padding=1
                )
            )

            in_channels = current_channels

            if i % 2 == 1:
                current_channels *= 2

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        x = self.pool(x).squeeze(-1).squeeze(-1)

        return self.fc(x)
