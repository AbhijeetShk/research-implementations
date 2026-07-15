import torch

from torch.utils.data import Dataset

from src.preprocessing.sample_builder import (
    build_exominer_sample
)


class ExoMinerDataset(Dataset):

    def __init__(
        self,
        candidates
    ):
        self.candidates = candidates

    def __len__(self):
        return len(
            self.candidates
        )

    def __getitem__(
        self,
        idx
    ):
        candidate = (
            self.candidates[idx]
        )

        sample = (
            build_exominer_sample(
                time=candidate["time"],
                flux=candidate["flux"],
                centroid_x=candidate["centroid_x"],
                centroid_y=candidate["centroid_y"],
                period=candidate["period"],
                epoch=candidate["epoch"],
                duration=candidate["duration"]
            )
        )

        return {
            "full_flux": torch.tensor(
                sample["full_flux"],
                dtype=torch.float32
            ).unsqueeze(0),

            "transit_flux": torch.tensor(
                sample["transit_flux"],
                dtype=torch.float32
            ).unsqueeze(0),

            "full_centroid": torch.tensor(
                sample["full_centroid"],
                dtype=torch.float32
            ).unsqueeze(0),

            "transit_centroid": torch.tensor(
                sample["transit_centroid"],
                dtype=torch.float32
            ).unsqueeze(0),

            "scalar_features": torch.tensor(
                sample["scalar_features"],
                dtype=torch.float32
            ),

            "label": torch.tensor(
                candidate["label"],
                dtype=torch.float32
            )
        }