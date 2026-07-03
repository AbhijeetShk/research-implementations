import torch
from tqdm import tqdm


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):
    model.train()

    running_loss = 0.0

    for batch in tqdm(loader):

        full_flux = (
            batch["full_flux"]
            .to(device)
        )

        transit_flux = (
            batch["transit_flux"]
            .to(device)
        )

        full_centroid = (
            batch["full_centroid"]
            .to(device)
        )

        transit_centroid = (
            batch["transit_centroid"]
            .to(device)
        )

        scalar_features = (
            batch["scalar_features"]
            .to(device)
        )

        labels = (
            batch["label"]
            .float()
            .unsqueeze(1)
            .to(device)
        )

        outputs = model(
            full_flux,
            transit_flux,
            full_centroid,
            transit_centroid,
            scalar_features
        )

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return (
        running_loss
        / len(loader)
    )