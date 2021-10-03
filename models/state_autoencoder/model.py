import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AE(nn.Module):
    def __init__(self, in_channel=4):
        super().__init__()
        self.encoder1 = nn.Conv1d(
            in_channel, 8, 3
        )
        self.encoder1_pool = nn.MaxPool1d(
            3, stride=2
        )
        self.encoder2 = nn.Conv1d(
            8, 16, 3
        )
        self.encoder2_pool = nn.MaxPool1d(
            3, stride=2
        )
        self.decoder1 = nn.ConvTranspose1d(
            16, 8, 5, stride=2
        )
        self.decoder2 = nn.ConvTranspose1d(
            8, 4, 5, stride=2
        )
        self.decoder3 = nn.ConvTranspose1d(
            4, 1, 3, stride=1
        )

    def forward(self, features):
        activation = self.encoder1(features)
        activation = self.encoder1_pool(activation)
        activation = torch.relu(activation)

        activation = self.encoder2(activation)
        activation = self.encoder2_pool(activation)
        activation = torch.relu(activation)

        reconstructed = self.decoder1(activation)
        reconstructed = self.decoder2(reconstructed)
        reconstructed = self.decoder3(reconstructed)
        return reconstructed