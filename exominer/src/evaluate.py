import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np

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

@torch.no_grad()
def evaluate_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        full_flux = batch["full_flux"].to(device)
        transit_flux = batch["transit_flux"].to(device)
        full_centroid = batch["full_centroid"].to(device)
        transit_centroid = batch["transit_centroid"].to(device)
        scalar_features = batch["scalar_features"].to(device)
        labels = batch["label"].float().unsqueeze(1).to(device)

        outputs = model(
            full_flux, transit_flux, 
            full_centroid, transit_centroid, 
            scalar_features
        )
        probs = torch.sigmoid(outputs)

        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # if single class is present in batch subset then handle that case for roc_auc_score
    try:
        roc_auc = roc_auc_score(labels, preds)
    except ValueError:
        roc_auc = float('nan')

    precision, recall, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)

    binary_preds = (preds > 0.5).astype(float)
    accuracy = np.mean(binary_preds == labels)

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }