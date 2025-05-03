import torch
import json
from torchvision import transforms
from PIL import Image
import os

class Predictor:
    def __init__(self, model_path="model.pth"):
        self.model = SimpleCNN()  # Réutiliser la classe du modèle
        self.model.load_state_dict(torch.load(model_path))
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)
        return output.argmax().item()  # Retourne la classe prédite

# Exemple d'utilisation
if __name__ == "__main__":
    predictor = Predictor()
    image_dir = "data/test"
    predictions = {}
    for case_id in os.listdir(image_dir):
        image_path = os.path.join(image_dir, case_id, "image.tiff")
        pred = predictor.predict(image_path)
        predictions[case_id] = {"inflammatory-cells": pred}  # Format simplifié

    # Sauvegarde des prédictions pour l'évaluation
    with open("evaluation/test/output/detected-inflammatory-cells.json", "w") as f:
        json.dump(predictions, f)