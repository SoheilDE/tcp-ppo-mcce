from mlagents.torch_utils import torch, nn


class TemporalCNNEncoder(nn.Module):
    def __init__(self, window_size=20, channels=(32, 64), hidden_units=128):
        super().__init__()
        self.window_size = window_size
        self.layers = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels[1], hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
        )

    def forward(self, temporal_window: torch.Tensor) -> torch.Tensor:
        return self.layers(temporal_window.unsqueeze(1))
