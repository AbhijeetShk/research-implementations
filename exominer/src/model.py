import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import get_output_dim


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
    def __init__(
        self,
        in_channels=1,
        channels=[16, 32, 64]
    ):
        super().__init__()

        layers = []

        current_channels = in_channels

        for out_channels in channels:
            layers.append(
                ConvBlock(
                    current_channels,
                    out_channels
                )
            )

            current_channels = out_channels

        self.features = nn.Sequential(*layers)

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

 
    # Allows the primary flux representation to dynamically attend 
    # to spatial centroid evidence, filtering out noisy photocenter shifts.

class CrossModalAttentionFusion(nn.Module):

    def __init__(self, flux_dim, centroid_dim):
        super().__init__()
        self.query_proj = nn.Linear(flux_dim, flux_dim)
        self.key_proj = nn.Linear(centroid_dim, flux_dim)     #centroid dim -> flux dim
        self.value_proj = nn.Linear(centroid_dim, flux_dim)
        self.scale = flux_dim ** -0.5

    def forward(self, flux_feat, centroid_feat):
        Q = self.query_proj(flux_feat).unsqueeze(1)    # [B, 1, D] for Q,K,V
        K = self.key_proj(centroid_feat).unsqueeze(1)
        V = self.value_proj(centroid_feat).unsqueeze(1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale # [B, 1, 1]
        attn_weights = F.softmax(scores, dim=-1)
        
        attended_centroid = torch.matmul(attn_weights, V).squeeze(1) # [B, D_flux]
        
        # residual fusion with original flux features
        return flux_feat + attended_centroid
    
class MultiBranchExoMiner(nn.Module):
    def __init__(self, scalar_dim, full_flux_len=200, transit_flux_len=50, full_centroid_len=50, transit_centroid_len=50):
        super().__init__()

        self.full_flux_branch = TimeSeriesCNN(channels=[16, 32, 64])
        self.transit_flux_branch = TimeSeriesCNN(channels=[16, 32, 64])
        self.full_centroid_branch = TimeSeriesCNN(channels=[16, 32, 64])
        self.transit_centroid_branch = TimeSeriesCNN(channels=[16, 32, 64])

        self.scalar_encoder = ScalarFeatureEncoder(scalar_dim)

        flux_feat_dim = get_output_dim(self.full_flux_branch, input_length=full_flux_len)
        transit_flux_feat_dim = get_output_dim(self.transit_flux_branch, input_length=transit_flux_len)
        full_centroid_feat_dim = get_output_dim(self.full_centroid_branch, input_length=full_centroid_len)
        transit_centroid_feat_dim = get_output_dim(self.transit_centroid_branch, input_length=transit_centroid_len)
        #Cross-modal attention modules for flux and centroid features(global and local streams)
        self.global_attention = CrossModalAttentionFusion(flux_feat_dim, full_centroid_feat_dim)
        self.local_attention = CrossModalAttentionFusion(transit_flux_feat_dim, transit_centroid_feat_dim)
        
        scalar_out_dim = 128 

        # fusion_dim = (
        #     flux_feat_dim + 
        #     transit_flux_feat_dim + 
        #     full_centroid_feat_dim + 
        #     transit_centroid_feat_dim + 
        #     scalar_out_dim
        # )
        # Fusion dimension after attention combination:
        # global_fused + local_fused + scalar_features
        fusion_dim = flux_feat_dim + transit_flux_feat_dim + scalar_out_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.PReLU(),
            nn.Linear(1024, 256),
            nn.PReLU(),
            nn.Linear(256, 1)
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
    # Apply Cross-Modal Attention (Flux queries Centroid)
        fused_global = self.global_attention(full_flux_features, full_centroid_features)
        fused_local = self.local_attention(transit_flux_features, transit_centroid_features)

        combined = torch.cat(
            [
                fused_global,
                fused_local,
                scalar_features
            ],
            dim=1
        )

        return self.classifier(combined)    