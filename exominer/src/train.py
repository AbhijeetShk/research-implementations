import torch
from tqdm import tqdm
from src.checkpoint import save_checkpoint

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Training"):
        full_flux = batch["full_flux"].to(device)
        transit_flux = batch["transit_flux"].to(device)
        full_centroid = batch["full_centroid"].to(device)
        transit_centroid = batch["transit_centroid"].to(device)
        scalar_features = batch["scalar_features"].to(device)
        labels = batch["label"].float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(
            full_flux, transit_flux, 
            full_centroid, transit_centroid, 
            scalar_features
        )
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0

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
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


def fit(model, train_loader, val_loader, optimizer, criterion, epochs, device, checkpoint_path="best_model.pt"):
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                path=checkpoint_path
            )
            print("Saved Best Model (Validation-Driven)")

        print("-" * 50)