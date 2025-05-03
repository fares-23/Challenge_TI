from torch.utils.data import Dataset
from PIL import Image
import os
import json

class MonkeyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        annotations = {}
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                case_id = filename.split("_")[0]
                with open(os.path.join(self.data_dir, filename)) as f:
                    annotations[case_id] = json.load(f)
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        case_id = list(self.annotations.keys())[idx]
        image_path = os.path.join(self.data_dir, f"{case_id}.tiff")
        image = Image.open(image_path).convert("RGB")
        label = self.annotations[case_id]["label"]  # À adapter selon le format réel
        if self.transform:
            image = self.transform(image)
        return image, label