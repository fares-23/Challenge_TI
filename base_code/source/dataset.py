# Importation des modules nécessaires
import os  # Pour interagir avec le système de fichiers
import json  # Pour lire les fichiers JSON
from PIL import Image  # Pour manipuler les images
import torch  # Framework pour le calcul tensoriel
from torch.utils.data import Dataset  # Classe de base pour les datasets PyTorch
from torchvision import transforms  # Pour appliquer des transformations sur les images


# Définition d'une classe personnalisée pour le dataset
class MonkeyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Initialise le dataset.
        :param data_dir: Chemin vers le répertoire contenant les données (images et annotations).
        :param transform: Transformations à appliquer aux images (par défaut : redimensionnement et conversion en tenseur).
        """
        self.data_dir = data_dir  # Répertoire des données
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),  # Redimensionne les images à 256x256 pixels
            transforms.ToTensor(),  # Convertit les images en tenseurs PyTorch
        ])
        self.annotations = self._load_annotations()  # Charge les annotations depuis les fichiers JSON

    def _load_annotations(self):
        """
        Charge les annotations depuis les fichiers JSON dans le répertoire des données.
        :return: Dictionnaire où les clés sont les IDs des cas et les valeurs sont les annotations.
        """
        annotations = {}
        for file in os.listdir(self.data_dir):  # Parcourt tous les fichiers du répertoire
            if file.endswith(".json"):  # Filtre les fichiers JSON
                case_id = file.split("_")[0]  # Extrait l'ID du cas à partir du nom du fichier
                with open(os.path.join(self.data_dir, file)) as f:  # Ouvre le fichier JSON
                    annotations[case_id] = json.load(f)  # Charge le contenu JSON dans le dictionnaire
        return annotations

    def __len__(self):
        """
        Retourne le nombre d'éléments dans le dataset.
        :return: Nombre d'annotations (et donc d'images associées).
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Retourne un élément du dataset (image et label) à l'index donné.
        :param idx: Index de l'élément à récupérer.
        :return: Tuple (image transformée, label sous forme de tenseur).
        """
        case_id = list(self.annotations.keys())[idx]  # Récupère l'ID du cas correspondant à l'index
        img_path = os.path.join(self.data_dir, f"{case_id}.tiff")  # Chemin vers l'image associée
        image = Image.open(img_path).convert("RGB")  # Ouvre l'image et la convertit en RGB
        label = self._get_label(self.annotations[case_id])  # Récupère le label à partir des annotations
        return self.transform(image), torch.tensor(label, dtype=torch.long)  # Applique les transformations et retourne l'image et le label

    def _get_label(self, annotation):
        """
        Convertit les annotations en label numérique.
        :param annotation: Annotation brute (dictionnaire chargé depuis le JSON).
        :return: Label numérique (exemple : 0=lymphocyte, 1=monocyte, etc.).
        """
        match(annotation["inflammatory-cells"]):
            
            case "lymphocyte":
                return 0
            case "monocyte":
                return 1
            case "inflammatory-cells":
                return 2
            case _:
                return -1  # Placeholder : à adapter selon les classes spécifiques du dataset
            
print("Dataset loaded successfully.")