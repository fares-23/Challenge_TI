import os
import json
import numpy as np
import torch
import cv2
import re


def load_image(file_id, image_dir):
    """Charge une image RGB .npy depuis un ID de fichier"""
    path = os.path.join(image_dir, f"{file_id}.npy")
    return np.load(path)


def load_mask(file_id, mask_dir):
    """Charge un masque (ex: segmentation ou centre) depuis un ID"""
    path = os.path.join(mask_dir, f"{file_id}.npy")
    return np.load(path)


def load_nuclick_annotation(file_id, mask_dir):
    """
    Charge un fichier .npy NuClick (6 ou 9 canaux) et retourne les masques utiles
    """
    data = np.load(os.path.join(mask_dir, f"{file_id}.npy")).astype(np.uint8)
    if data.shape[-1] == 6:
        return {
            "binary_mask": data[:, :, 3],
            "class_mask": data[:, :, 4],
            "contour_mask": data[:, :, 5]
        }
    else:  # version 9 canaux
        return {
            "inflamm_mask": data[:, :, 3],
            "inflamm_contour_mask": data[:, :, 4],
            "lymph_mask": data[:, :, 5],
            "lymph_contour_mask": data[:, :, 6],
            "mono_mask": data[:, :, 7],
            "mono_contour_mask": data[:, :, 8]
        }


def load_json_annotation(file_id, json_dir):
    """Charge un fichier .json contenant des annotations"""
    path = os.path.join(json_dir, f"{file_id}.json")
    with open(path, "r") as f:
        return json.load(f)


def get_all_file_ids(image_dir):
    """Retourne les noms de fichiers sans extension"""
    return [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".npy")]


def extract_id(file_name):
    """Extrait un ID unique d’un nom de fichier image, ex: 'A_P1234_PAS.tif' -> 'A_P1234'"""
    match = re.match(r"([A-Z]_P\d+)_", file_name, re.IGNORECASE)
    return match.group(1) if match else None


def centre_split(file_ids, val_fold=1):
    """Découpe les fichiers en train/val selon les centres (A, B, C, D)"""
    centres = ["A", "B", "C", "D"]
    test_centre = centres[val_fold - 1]

    train_file_ids = []
    val_file_ids = []

    for id in file_ids:
        if id[0] != test_centre:
            train_file_ids.append(id)
        else:
            val_file_ids.append(id)

    split = {
        "train_file_ids": train_file_ids,
        "val_file_ids": val_file_ids,
    }
    return split


def imagenet_normalise(img):
    """Normalise une image RGB numpy [H,W,3] avec les stats d'ImageNet"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (img - mean) / std


def imagenet_denormalise(img):
    """Inverse la normalisation ImageNet"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (img * std) + mean


def imagenet_normalise_torch(img):
    """Normalisation pour tenseur torch [B,3,H,W]"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
    return (img - mean) / std


def extract_centroids(mask, area_threshold=3):
    """Extrait les centroïdes d'un masque binaire (instances)"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for c in contours:
        if cv2.contourArea(c) < area_threshold:
            continue
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
    return centroids

import numpy as np
import cv2
import torch
import json
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max


def draw_disks(canvas_size, centroids, radius=5):
    """
    Dessine des disques autour des centroïdes sur une image vide.
    """
    mask = np.zeros(canvas_size, dtype=np.uint8)
    for x, y in centroids:
        if radius == 0:
            mask[y, x] = 1
        else:
            cv2.circle(mask, (x, y), radius, 255, -1)
    return (mask > 0).astype(np.float32)


def dilate_mask(mask, radius=5):
    """
    Dilate un masque binaire avec un élément structurant circulaire.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def generate_distance_map(mask):
    """
    Retourne la distance transformée d'un masque binaire.
    """
    return distance_transform_edt(mask)


def generate_regression_map(binary_mask, d_thresh=5, alpha=3, scale=3):
    dist = distance_transform_edt(binary_mask == 0)
    M = (np.exp(alpha * (1 - dist / d_thresh)) - 1) / (np.exp(alpha) - 1)
    M[M < 0] = 0
    return M * scale


def px_to_mm(px, mpp=0.24199951445730394):
    """
    Conversion pixels vers millimètres.
    """
    return px * mpp / 1000


def extract_dotmaps(prob_map, min_dist=5, threshold=0.5, method="local_max"):
    """
    Post-traitement d'une carte de probabilité pour obtenir les centroïdes.
    """
    if method == "local_max":
        coords = peak_local_max(prob_map, min_distance=min_dist, threshold_abs=threshold)
        return coords[:, [1, 0]].tolist()  # x, y
    elif method == "threshold":
        binary = (prob_map > threshold).astype(np.uint8)
        _, _, _, centroids = cv2.connectedComponentsWithStats(binary)
        return np.rint(centroids[1:]).astype(int).tolist()
    else:
        raise ValueError("Méthode inconnue : choisissez 'local_max' ou 'threshold'")


def scale_coords(coords, factor):
    return [[int(x * factor), int(y * factor)] for x, y in coords]


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def collate_fn(batch):
    return torch.as_tensor(np.array(batch), dtype=torch.float)
