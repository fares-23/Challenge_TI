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


def filter_points_with_mask(points, mask, point_mpp=0.24, mask_mpp=8.0, margin=1):
    """
    Filtre les points situés en dehors du masque binaire.

    Args:
        points: liste de dicts [{'x','y','type','prob'}]
        mask: masque binaire (0: fond, 1: tissu)
        point_mpp: résolution (µm/px) des points
        mask_mpp: résolution (µm/px) du masque
        margin: taille de la fenêtre autour du point pour vérification

    Returns:
        liste filtrée de points valides
    """
    scale = mask_mpp / point_mpp
    valid_points = []

    for pt in points:
        x_px = int(np.round(pt["x"] / scale))
        y_px = int(np.round(pt["y"] / scale))

        # Zone autour du point
        coords = [
            (x_px + dx, y_px + dy)
            for dx in [-margin, 0, margin]
            for dy in [-margin, 0, margin]
        ]
        coords = [(x, y) for x, y in coords if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]]
        count_inside = sum(mask[y, x] for x, y in coords)

        if count_inside / len(coords) >= 0.5:
            valid_points.append(pt)

    return valid_points


def normalize_probs(points, min_prob=0.5):
    """
    Normalise les scores de confiance des détections à [min_prob, 1.0].
    """
    scores = [p["prob"] for p in points]
    min_score = min(scores)
    max_score = max(scores)
    norm_points = []

    for p in points:
        norm = (p["prob"] - min_score) / (max_score - min_score + 1e-8)
        p["prob"] = norm * (1 - min_prob) + min_prob
        norm_points.append(p)

    return norm_points


def non_max_suppression(boxes, threshold=0.5):
    """
    Suppression non maximale (NMS) pour filtrer les boîtes qui se chevauchent trop.

    Args:
        boxes: tableau de forme [N, 5] : x1, y1, x2, y2, score
        threshold: seuil de recouvrement (IoU)

    Returns:
        liste d’indices des boîtes retenues
    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(float)
    x1, y1, x2, y2, scores = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:][iou <= threshold]

    return keep

import numpy as np


def nms(boxes: np.ndarray, threshold: float = 0.5):
    """
    Applique la suppression non maximale (NMS) sur des boîtes prédictives.

    Args:
        boxes: tableau [N, 5] avec (x1, y1, x2, y2, score)
        threshold: seuil de recouvrement (IoU) à supprimer

    Returns:
        indices des boîtes conservées
    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(float)
    x1, y1, x2, y2, scores = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # du + confiant au - confiant

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        # Intersection avec les autres boîtes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:][iou <= threshold]

    return keep


def point_to_box(x, y, size, prob=None):
    """
    Transforme un point (x,y) en boîte carrée centrée sur ce point.

    Args:
        x, y: coordonnées du centre
        size: demi-taille de la boîte (boîte = 2*size)
        prob: optionnel, score de confiance

    Returns:
        boîte (x1, y1, x2, y2, [prob])
    """
    box = [x - size, y - size, x + size, y + size]
    if prob is not None:
        box.append(prob)
    return np.array(box)


def get_center_from_box(box, offset):
    """
    Calcule le centre d'une boîte à partir d'un coin haut-gauche et d’un rayon.

    Args:
        box: [x1, y1, x2, y2]
        offset: (int) = demi-taille utilisée

    Returns:
        tuple (x, y)
    """
    return (box[0] + offset, box[1] + offset)



def apply_nms_in_tiles(
    detection_points: list[dict],
    image_shape: tuple,
    tile_size: int = 512,
    box_size: int = 5,
    overlap_thresh: float = 0.5,
):
    """
    Applique la suppression non maximale (NMS) par tuiles sur une grande image.

    Args:
        detection_points: liste de dicts {'x','y','type','prob'}
        image_shape: (H, W)
        tile_size: taille d'une tuile
        box_size: demi-taille des boîtes autour des points
        overlap_thresh: seuil IoU pour suppression

    Returns:
        Liste filtrée des points après NMS
    """
    H, W = image_shape
    final_points = []

    for y0 in range(0, H, tile_size):
        for x0 in range(0, W, tile_size):
            x1, y1 = x0 + tile_size, y0 + tile_size

            # Points dans la tuile
            tile_points = [
                p for p in detection_points
                if x0 <= p["x"] < x1 and y0 <= p["y"] < y1
            ]

            if len(tile_points) < 2:
                final_points.extend(tile_points)
                continue

            boxes = np.array([
                [p["x"] - box_size, p["y"] - box_size,
                 p["x"] + box_size, p["y"] + box_size, p["prob"]]
                for p in tile_points
            ])
            keep = nms(boxes, threshold=overlap_thresh)
            final_points.extend([tile_points[i] for i in keep])

    return final_points


def erode_mask(mask: np.ndarray, size: int = 3, iterations: int = 1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)


def morph_postprocess(mask: np.ndarray, size: int = 3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def add_background_channel(mask: np.ndarray):
    """
    Ajoute un canal de fond pour la segmentation multi-classe.

    Args:
        mask: [C, H, W]

    Returns:
        mask_out: [C+1, H, W]
    """
    C, H, W = mask.shape
    new_mask = np.zeros((C + 1, H, W), dtype=np.uint8)
    union = np.any(mask, axis=0)
    new_mask[0] = ~union
    new_mask[1:] = mask
    return new_mask.astype(np.uint8)
