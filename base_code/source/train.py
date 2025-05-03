import torch
from torch.utils.data import DataLoader
from dataset import MonkeyDataset
from model import HybridModel
import torch.optim as optim

DATA_DIR_CHRISTELLE = r"C:\Users\Christelle\Documents\CHALLENGE\images"
# DATA_DIR_ADAM = 
TRAINED_MODEL_PATH = r"C:\Users\Christelle\Documents\CHALLENGE\Challenge_TI\base_code\trained_model"

def train():
    # Config
    data_dir = DATA_DIR_CHRISTELLE  # Chemin vers le répertoire contenant les images et annotations
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
    torch.save(model.state_dict(), TRAINED_MODEL_PATH)

if __name__ == "__main__":
    train()
    print("Training completed and model saved as model.pth")