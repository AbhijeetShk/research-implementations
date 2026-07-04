import torch


@torch.no_grad()
def evaluate(
    model,
    loader,
    device
):
    model.eval()

    correct = 0
    total = 0

    for batch in loader:

        full_flux = batch["full_flux"].to(device)

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

        probs = torch.sigmoid(
            outputs
        )

        preds = (
            probs > 0.5
        ).float()

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

    return correct / total