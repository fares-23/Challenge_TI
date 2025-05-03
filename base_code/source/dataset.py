import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MonkeyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        annotations = {}
        for file in os.listdir(self.data_dir):
            if file.endswith(".json"):
                case_id = file.split("_")[0]
                with open(os.path.join(self.data_dir, file)) as f:
                    annotations[case_id] = json.load(f)
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        case_id = list(self.annotations.keys())[idx]
        img_path = os.path.join(self.data_dir, f"{case_id}.tiff")
        image = Image.open(img_path).convert("RGB")
        label = self._get_label(self.annotations[case_id])
        return self.transform(image), torch.tensor(label, dtype=torch.long)

    def _get_label(self, annotation):
        # Convertit les annotations en label numérique (ex: 0=lymphocyte, 1=monocyte, etc.)
        return 0  # À adapter selon vos classes