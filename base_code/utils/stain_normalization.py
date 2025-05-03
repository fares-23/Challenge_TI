import cv2
import numpy as np

def normalize_stain(image_path):
    """Normalisation des couleurs avec la méthode Macenko (simplifiée)."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Exemple simplifié : ajustement de la luminosité
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_norm = clahe.apply(l)
    lab_norm = cv2.merge((l_norm, a, b))
    return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2RGB)