import torch
from tqdm import tqdm

from src.checkpoint import save_checkpoint


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):
    model.train()

    running_loss = 0.0

    for batch in tqdm(
        loader,
        desc="Training"
    ):

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

    avg_loss = (
        running_loss
        / len(loader)
    )

    return avg_loss


def fit(
    model,
    loader,
    optimizer,
    criterion,
    epochs,
    device
):
    best_loss = float("inf")

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch+1}/{epochs}]"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        if train_loss < best_loss:

            best_loss = train_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                path="best_model.pt"
            )

            print(
                "Saved Best Model"
            )

        print("-" * 50)