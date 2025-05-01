import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.simple_cnn import SimpleCellDetector
from src.dataset import CustomMaskDataset
from src.config import SimpleTrainingConfig


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    config = SimpleTrainingConfig(dataset_dir="data")
    dataset = CustomMaskDataset(config.image_dir, config.mask_dir)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCellDetector().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"[Epoch {epoch+1}] Training Loss: {train_loss:.4f}")

    # Sauvegarde du modèle
    torch.save(model.state_dict(), "models/simple_cell_detector.pth")
    print("✅ Modèle sauvegardé.")


if __name__ == "__main__":
    main()
