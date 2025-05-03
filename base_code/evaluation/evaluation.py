import json
import os
import numpy as np
from sklearn.metrics import auc

def calculate_froc(gt_dir, pred_dir, thresholds=np.linspace(0.1, 1.0, 10)):
    # Chargement des annotations et prédictions
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(".json")]
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith(".json")]

    sensitivities, fp_per_image = [], []

    for threshold in thresholds:
        tp, total_gt, fp = 0, 0, 0

        for gt_file in gt_files:
            case_id = gt_file.split("_")[0]
            with open(os.path.join(gt_dir, gt_file)) as f:
                gt_data = json.load(f)
            
            pred_path = os.path.join(pred_dir, f"detected_{case_id}_inflammatory-cells.json")
            if not os.path.exists(pred_path):
                continue

            with open(pred_path) as f:
                pred_data = json.load(f)

            # Logique de matching simplifiée
            total_gt += len(gt_data["inflammatory-cells"])
            for pred in pred_data.get("detections", []):
                if pred["confidence"] >= threshold:
                    if self._is_match(pred, gt_data["inflammatory-cells"]):
                        tp += 1
                    else:
                        fp += 1

        sensitivity = tp / max(total_gt, 1)
        avg_fp = fp / max(len(gt_files), 1)
        sensitivities.append(sensitivity)
        fp_per_image.append(avg_fp)

    return {
        "froc_auc": auc(fp_per_image, sensitivities),
        "sensitivities": sensitivities,
        "fp_per_image": fp_per_image
    }

def _is_match(self, pred, gt_list, distance_threshold=10):
    for gt in gt_list:
        if ((pred["x"] - gt["x"])**2 + (pred["y"] - gt["y"])**2) <= distance_threshold**2:
            return True
    return False

if __name__ == "__main__":
    metrics = calculate_froc(
        gt_dir="evaluation/ground_truth",
        pred_dir="evaluation/test/output"
    )
    with open("evaluation/test/output/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)