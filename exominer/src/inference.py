import torch
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def predict_with_uncertainty(model: torch.nn.Module, dataloader, device: torch.device, num_samples: int = 30):
#   dropout during inference to compute epistemic uncertainty: Monte Carlo sampling.

    model.train()  #dropout layers active during evaluation
    
    all_mean_probs = []
    all_uncertainties = []
    all_labels = []

    for batch in dataloader:
        full_flux = batch["full_flux"].to(device)
        transit_flux = batch["transit_flux"].to(device)
        full_centroid = batch["full_centroid"].to(device)
        transit_centroid = batch["transit_centroid"].to(device)
        scalar_features = batch["scalar_features"].to(device)
        labels = batch["label"].float().unsqueeze(1).to(device)

        # collecting T forward passes with dropout enabled
        monte_carlo_preds = []
        for _ in range(num_samples):
            outputs = model(
                full_flux, transit_flux, 
                full_centroid, transit_centroid, 
                scalar_features
            )
            probs = torch.sigmoid(outputs)
            monte_carlo_preds.append(probs.cpu().numpy())

        # Shape: [num_samples, batch_size, 1]
        preds_stack = np.stack(monte_carlo_preds, axis=0)
        
        # Mean prediction and standard deviation (epistemic uncertainty)
        mean_probs = preds_stack.mean(axis=0)
        uncertainty = preds_stack.std(axis=0)

        all_mean_probs.append(mean_probs)
        all_uncertainties.append(uncertainty)
        all_labels.append(labels.cpu().numpy())

    return np.vstack(all_mean_probs), np.vstack(all_uncertainties), np.vstack(all_labels)