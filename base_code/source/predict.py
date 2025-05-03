import os
import json
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from torchvision import transforms
from model import HybridModel
from torchvision.models import ResNet50_Weights

#DATA_DIR_CHRISTELLE = r"C:\Users\Christelle\Documents\CHALLENGE\images"
#DATA_DIR_ADAM = r"/Volumes/TOSHIBA/CHALLENGE/images"
class Predictor:
    def __init__(self):
        weights = ResNet50_Weights.DEFAULT
        self.model = HybridModel(weights)
        # self.model.load_state_dict(torch.load(model_path))
        # self.transform = transforms.Compose([
        #     transforms.Resize((256, 256)),
        #     transforms.ToTensor(),
        # ])
        self.preprocess = weights.transforms()

    def predict(self, image_dir, output_dir="evaluation/test/output"):
        os.makedirs(output_dir, exist_ok=True)
        predictions = {}
        img_cpt = 0
        
        for case_id in os.listdir(image_dir):
            case_path = os.path.join(image_dir, case_id)
            if not os.path.isdir(case_path):
                continue
            for image_name in os.listdir(case_path):
                if image_name.endswith(".tif") or image_name.endswith(".tiff"):
                    img_path = os.path.join(case_path, image_name)
                    break
            # img_path = os.path.join(image_dir, case_id, "A_P000001_IHC_CPG.tif")
            image = Image.open(img_path).convert("RGB")
            # image = self.transform(image).unsqueeze(0)
            if image.size[0] * image.size[1] >= 178956970:  # Check if the image exceeds the limit
                print(f"Image {case_id} is too large, skipping.")
            image = self.preprocess(image).unsqueeze(0)
                        
            with torch.no_grad():
                output = self.model(image)
                pred_class = output.argmax().item()
            
            predictions[case_id] = {"inflammatory-cells": pred_class}
            image.close()
            img_cpt += 1
            if img_cpt >= 2:
                break
        # Sauvegarde au format attendu par le challenge
        with open(f"{output_dir}/detected-inflammatory-cells.json", "w") as f:
            json.dump(predictions, f)

if __name__ == "__main__":
    predictor = Predictor()
    #predictor.predict(DATA_DIR_CHRISTELLE)
    predictor.predict(DATA_DIR_ADAM)

    print("Prediction completed and saved to evaluation/test/output/detected-inflammatory-cells.json")

J'utilise ce code et j'ai cette erreur : 
HybridModel loaded successfully.
TIFFFillTile: 0: Invalid tile byte count, tile 0.
Traceback (most recent call last):
  File "/Users/adamakil/Desktop/Traitement des Images/challenge/Challenge_TI/base_code/source/predict.py", line 59, in <module>
    predictor.predict(DATA_DIR_ADAM)
  File "/Users/adamakil/Desktop/Traitement des Images/challenge/Challenge_TI/base_code/source/predict.py", line 37, in predict
    image = Image.open(img_path).convert("RGB")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/PIL/Image.py", line 995, in convert
    self.load()
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/PIL/TiffImagePlugin.py", line 1237, in load
    return self._load_libtiff()
           ^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/PIL/TiffImagePlugin.py", line 1342, in _load_libtiff
    raise OSError(err)
OSError: -2