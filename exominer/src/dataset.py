import torch
from torch.utils.data import Dataset


class ExoMinerDataset(Dataset):

    def __init__(
        self,
        full_flux,
        transit_flux,
        full_centroid,
        transit_centroid,
        scalar_features,
        labels
    ):
        self.full_flux = full_flux
        self.transit_flux = transit_flux

        self.full_centroid = full_centroid
        self.transit_centroid = transit_centroid

        self.scalar_features = scalar_features

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sample = {
            "full_flux":
                torch.tensor(
                    self.full_flux[idx],
                    dtype=torch.float32
                ),

            "transit_flux":
                torch.tensor(
                    self.transit_flux[idx],
                    dtype=torch.float32
                ),

            "full_centroid":
                torch.tensor(
                    self.full_centroid[idx],
                    dtype=torch.float32
                ),

            "transit_centroid":
                torch.tensor(
                    self.transit_centroid[idx],
                    dtype=torch.float32
                ),

            "scalar_features":
                torch.tensor(
                    self.scalar_features[idx],
                    dtype=torch.float32
                ),

            "label":
                torch.tensor(
                    self.labels[idx],
                    dtype=torch.float32
                )
        }

        return sample