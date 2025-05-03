import torch
from torch.utils.data import DataLoader
from dataset import MonkeyDataset
from model import HybridModel
import torch.optim as optim

def train():
    # Config
    data_dir = "data/train"
    batch_size = 8
    epochs = 10

    # Chargement des données
    dataset = MonkeyDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Modèle et optimisation
    model = HybridModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Entraînement
    for epoch in range(epochs):
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Sauvegarde du modèle
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    train()