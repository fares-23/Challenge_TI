import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model import HybridModel

class Predictor:
    def __init__(self, model_path="model.pth"):
        self.model = HybridModel()
        self.model.load_state_dict(torch.load(model_path))
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def predict(self, image_dir, output_dir="evaluation/test/output"):
        os.makedirs(output_dir, exist_ok=True)
        predictions = {}
        
        for case_id in os.listdir(image_dir):
            img_path = os.path.join(image_dir, case_id, "image.tiff")
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(image)
                pred_class = output.argmax().item()
            
            predictions[case_id] = {"inflammatory-cells": pred_class}

        # Sauvegarde au format attendu par le challenge
        with open(f"{output_dir}/detected-inflammatory-cells.json", "w") as f:
            json.dump(predictions, f)

if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict("data/test")