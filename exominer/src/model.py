import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=5,
                padding=2,
            ),
            nn.PReLU(),

            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=5,
                padding=2,
            ),
            nn.PReLU(),

            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)


class TimeSeriesCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x

class DualFluxNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.full_flux_branch = TimeSeriesCNN()
        self.transit_flux_branch = TimeSeriesCNN()

        self.classifier = nn.Sequential(
            nn.Linear(3200, 512),
            nn.PReLU(),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, full_orbit_flux, transit_view_flux):

        full_features = self.full_flux_branch(
            full_orbit_flux
        )

        transit_features = self.transit_flux_branch(
            transit_view_flux
        )

        combined = torch.cat(
            [full_features, transit_features],
            dim=1
        )

        output = self.classifier(combined)

        return output

class ScalarFeatureEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.PReLU(),

            nn.Linear(128, 128),
            nn.PReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class MultiBranchExoMiner(nn.Module):
    def __init__(self, scalar_dim):
        super().__init__()


        self.full_flux_branch = TimeSeriesCNN()
        self.transit_flux_branch = TimeSeriesCNN()

        self.full_centroid_branch = TimeSeriesCNN()
        self.transit_centroid_branch = TimeSeriesCNN()

        self.scalar_encoder = ScalarFeatureEncoder(
            scalar_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                1600 * 4 + 128,
                1024
            ),
            nn.PReLU(),

            nn.Linear(
                1024,
                256
            ),
            nn.PReLU(),

            nn.Linear(
                256,
                1
            )
        )

    def forward(
        self,
        full_flux,
        transit_flux,
        full_centroid,
        transit_centroid,
        scalar_features
    ):
        full_flux_features = self.full_flux_branch(
            full_flux
        )

        transit_flux_features = self.transit_flux_branch(
            transit_flux
        )

        full_centroid_features = self.full_centroid_branch(
            full_centroid
        )

        transit_centroid_features = (
            self.transit_centroid_branch(
                transit_centroid
            )
        )

        scalar_features = self.scalar_encoder(
            scalar_features
        )

        combined = torch.cat(
            [
                full_flux_features,
                transit_flux_features,
                full_centroid_features,
                transit_centroid_features,
                scalar_features
            ],
            dim=1
        )

        return self.classifier(combined)    