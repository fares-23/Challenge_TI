"""
The following is a simple example evaluation method.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the evaluation, reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

import json

# for local debugging
import os
from pathlib import Path

import numpy as np
from monai.metrics import compute_froc_curve_data, compute_froc_score
from scipy.spatial import distance
from sklearn.metrics import auc

INPUT_DIRECTORY = Path(f"{os.getcwd()}/test/input")
OUTPUT_DIRECTORY = Path(f"{os.getcwd()}/test/output")
GROUND_TRUTH_DIRECTORY = Path(f"{os.getcwd()}/ground_truth")

# for docker building
# INPUT_DIRECTORY = Path("/input")
# OUTPUT_DIRECTORY = Path("/output")
# GROUND_TRUTH_DIRECTORY = Path("/opt/ml/input/data/ground_truth")

SPACING_LEVEL0 = 0.24199951445730394


def calculate_f1_metrics(tp: float, fn: float, fp: float) -> dict:
    """
    Calculate F1, Precision, Recall

    Args:
        tp
        fn
        fp
    Returns:
        metrics: {"F1", "Precision", "Recall"}
    """
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0

    if tp == 0 and fp == 0 and fn == 0:
        f1 = 0
    else:
        f1 = (2 * tp) / (2 * tp + fp + fn)

    return {
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
    }


def get_F1_scores(gt_dict, result_dict, radius: int):
    """
    Args:
        gt_dict (dict): Ground truth data containing points and regions of interest (ROIs).
        result_dict (dict): Result data containing detected points and their probabilities.
        radius (int): The maximum distance in pixels for considering a detection as a true positive.

    Returns:
        dict: A dictionary containing F1, Precison, Recall.
    """
    # create a mask from the gt coordinates with circles of given radius
    if len(gt_dict["points"]) == 0:
        return {
            "F1": 0,
            "Precision": 0,
            "Recall": 0,
        }
    gt_coords = [i["point"] for i in gt_dict["points"]]
    # result_prob = [i['probability'] for i in result_dict['points']]
    result_prob = [i["probability"] for i in result_dict["points"]]
    # make some dummy values between 0 and 1 for the result prob
    # result_prob = [np.random.rand() for i in range(len(result_dict['points']))]
    result_coords = [
        [i["point"][0], i["point"][1]] for i in result_dict["points"]
    ]

    # prepare the data for the FROC curve computation with monai
    (
        tp,
        fn,
        fp,
        tp_probs,
        fp_probs,
    ) = match_coordinates(
        gt_coords, result_coords, result_prob, radius
    )

    return calculate_f1_metrics(tp, fn, fp)


def get_froc_vals(gt_dict, result_dict, radius: int):
    """
    Computes the Free-Response Receiver Operating Characteristic (FROC) values for given ground truth and result data.
    Using https://docs.monai.io/en/0.5.0/_modules/monai/metrics/froc.html
    Args:
        gt_dict (dict): Ground truth data containing points and regions of interest (ROIs).
        result_dict (dict): Result data containing detected points and their probabilities.
        radius (int): The maximum distance in pixels for considering a detection as a true positive.

    Returns:
        dict: A dictionary containing FROC metrics such as sensitivity, false positives per mm²,
              true positive probabilities, false positive probabilities, total positives,
              area in mm², and FROC score.
    """
    # create a mask from the gt coordinates with circles of given radius
    if len(gt_dict["points"]) == 0:
        return {
            "sensitivity_slide": [0],
            "fp_per_mm2_slide": [0],
            "fp_probs_slide": [0],
            "tp_probs_slide": [0],
            "total_pos_slide": 0,
            "area_mm2_slide": 0,
            "froc_score_slide": 0,
        }
    gt_coords = [i["point"] for i in gt_dict["points"]]
    gt_rois = [i["polygon"] for i in gt_dict["rois"]]
    # compute the area of the polygon in roi
    area_mm2 = (
        SPACING_LEVEL0
        * SPACING_LEVEL0
        * gt_dict["area_rois"]
        / 1000000
    )
    # result_prob = [i['probability'] for i in result_dict['points']]
    result_prob = [i["probability"] for i in result_dict["points"]]
    # make some dummy values between 0 and 1 for the result prob
    # result_prob = [np.random.rand() for i in range(len(result_dict['points']))]
    result_coords = [
        [i["point"][0], i["point"][1]] for i in result_dict["points"]
    ]

    # prepare the data for the FROC curve computation with monai
    (
        true_positives,
        false_negatives,
        false_positives,
        tp_probs,
        fp_probs,
    ) = match_coordinates(
        gt_coords, result_coords, result_prob, radius
    )
    total_pos = len(gt_coords)
    # the metric is implemented to normalize by the number of images, we however want to have it by mm2, so we set
    # num_images = ROI area in mm2
    # fp_per_mm2_slide, sensitivity = compute_froc_curve_data(
    #     fp_probs, tp_probs, total_pos, area_mm2
    # )
    # if len(fp_per_mm2_slide) > 1 and len(sensitivity) > 1:
    #     area_under_froc = auc(fp_per_mm2_slide, sensitivity)
    #     froc_score = compute_froc_score(
    #         fp_per_mm2_slide,
    #         sensitivity,
    #         eval_thresholds=(10, 20, 50, 100, 200, 300),
    #     )
    # else:
    #     area_under_froc = 0
    #     froc_score = 0
    sensitivity, fp_per_mm2_slide, froc_score = get_froc_score(
        fp_probs, tp_probs, total_pos, area_mm2
    )
    return {
        "sensitivity_slide": list(sensitivity),
        "fp_per_mm2_slide": list(fp_per_mm2_slide),
        "fp_probs_slide": list(fp_probs),
        "tp_probs_slide": list(tp_probs),
        "total_pos_slide": total_pos,
        "area_mm2_slide": area_mm2,
        "froc_score_slide": float(froc_score),
    }


def get_froc_score(fp_probs, tp_probs, total_pos, area_mm2):
    eval_thresholds = (10, 20, 50, 100, 200, 300)

    fp_per_mm2, sensitivity = compute_froc_curve_data(
        fp_probs, tp_probs, total_pos, area_mm2
    )
    if len(fp_per_mm2) == 0 and len(sensitivity) == 0:
        return sensitivity, fp_per_mm2, 0
    if len(sensitivity) == 1:
        # we only have one true positive point, we have to compute the FROC values a bit differently
        sensitivity = [1]
        fp_per_mm2 = [len(fp_probs) / area_mm2]
        froc_score = np.mean(
            [int(fp_per_mm2[0] < i) for i in eval_thresholds]
        )
    else:
        # area_under_froc = auc(fp_per_mm2, sensitivity)
        froc_score = compute_froc_score(
            fp_per_mm2, sensitivity, eval_thresholds=eval_thresholds
        )

    return sensitivity, fp_per_mm2, froc_score


def match_coordinates(ground_truth, predictions, pred_prob, margin):
    """
    Matches predicted coordinates to ground truth coordinates within a certain distance margin
    and computes the associated probabilities for true positives and false positives.

    Args:
        ground_truth (list of tuples): List of ground truth coordinates as (x, y).
        predictions (list of tuples): List of predicted coordinates as (x, y).
        pred_prob (list of floats): List of probabilities associated with each predicted coordinate.
        margin (float): The maximum distance for considering a prediction as a true positive.

    Returns:
        true_positives (int): Number of correctly matched predictions.
        false_negatives (int): Number of ground truth coordinates not matched by any prediction.
        false_positives (int): Number of predicted coordinates not matched by any ground truth.
        tp_probs (list of floats): Probabilities of the true positive predictions.
        fp_probs (list of floats): Probabilities of the false positive predictions.
    """
    if len(ground_truth) == 0 and len(predictions) == 0:
        return 0, 0, 0, np.array([]), np.array([])
    # return true_positives, false_negatives, false_positives, np.array(tp_probs), np.array(fp_probs)
    if len(ground_truth) == 0 and len(predictions) != 0:
        return (
            0,
            0,
            len(predictions),
            np.array([]),
            np.array(pred_prob),
        )
    if len(ground_truth) != 0 and len(predictions) == 0:
        return 0, len(ground_truth), 0, np.array([]), np.array([])

    # Convert lists to numpy arrays for easier distance calculations
    gt_array = np.array(ground_truth)
    pred_array = np.array(predictions)
    pred_prob_array = np.array(pred_prob)

    # Distance matrix between ground truth and predictions
    dist_matrix = distance.cdist(gt_array, pred_array)

    # Initialize sets for matched indices
    matched_gt = set()
    matched_pred = set()

    # Iterate over the distance matrix to find the closest matches
    for gt_idx in range(len(ground_truth)):
        closest_pred_idx = np.argmin(dist_matrix[gt_idx])
        if dist_matrix[gt_idx, closest_pred_idx] <= margin:
            matched_gt.add(gt_idx)
            matched_pred.add(closest_pred_idx)
            dist_matrix[:, closest_pred_idx] = np.inf

    # Calculate true positives, false negatives, and false positives
    # print(f"preds {len(predictions)}")
    # print(f"gts {len(ground_truth)}")

    true_positives = len(matched_gt)
    false_negatives = len(ground_truth) - true_positives
    false_positives = len(predictions) - true_positives

    # Compute probabilities for true positives and false positives
    tp_probs = [pred_prob[i] for i in matched_pred]
    fp_probs = [
        pred_prob[i]
        for i in range(len(predictions))
        if i not in matched_pred
    ]

    return (
        true_positives,
        false_negatives,
        false_positives,
        np.array(tp_probs),
        np.array(fp_probs),
    )


def get_aggr_froc(metrics_dict):
    # https://docs.monai.io/en/0.5.0/_modules/monai/metrics/froc.html
    fp_probs = np.array(
        [
            item
            for sublist in metrics_dict["fp_probs_slide"]
            for item in sublist
        ]
    )
    tp_probs = np.array(
        [
            item
            for sublist in metrics_dict["tp_probs_slide"]
            for item in sublist
        ]
    )
    total_pos = sum(metrics_dict["total_pos_slide"])
    area_mm2 = sum(metrics_dict["area_mm2_slide"])

    # sensitivity, fp_overall = compute_froc_curve_data(fp_probs, tp_probs, total_pos, area_mm2)
    (
        fp_overall,
        sensitivity_overall,
    ) = compute_froc_curve_data(
        fp_probs, tp_probs, total_pos, area_mm2
    )
    if len(fp_overall) > 1 and len(sensitivity_overall) > 1:
        area_under_froc = auc(fp_overall, sensitivity_overall)
        froc_score = compute_froc_score(
            fp_overall,
            sensitivity_overall,
            eval_thresholds=(10, 20, 50, 100, 200, 300),
        )
    else:
        area_under_froc = 0
        froc_score = 0

    # return {'sensitivity_aggr': list(sensitivity_overall), 'fp_aggr': list(fp_overall),
    #         'fp_probs_aggr': list(fp_probs),
    #         'tp_probs_aggr': list(tp_probs), 'total_pos_aggr': total_pos, 'area_mm2_aggr': area_mm2,
    #         'froc_score_aggr': float(froc_score)}

    return {
        "area_mm2_aggr": area_mm2,
        "froc_score_aggr": float(froc_score),
    }


def format_metrics_for_aggr(metrics_list, cell_type):
    """
    Formats the metrics dictionary to be used in the aggregation function.
    """
    aggr = {}
    for d in [i[cell_type] for i in metrics_list]:
        # Iterate over each key-value pair in the dictionary
        for key, value in d.items():
            # If the key is not already in the collapsed_dict, initialize it with an empty list
            if key not in aggr:
                aggr[key] = []
            # Append the value to the list corresponding to the key
            aggr[key].append(value)

    return aggr


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    print(INPUT_DIRECTORY)
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(
        values=values, slug=slug
    )
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def convert_mm_to_pixel(data_dict, spacing=SPACING_LEVEL0):
    # Converts a distance in mm to pixels: coord in mm * 1000 * spacing
    points_pixels = []
    for d in data_dict["points"]:
        if len(d["point"]) == 2:
            d["point"] = [
                mm_to_pixel(d["point"][0]),
                mm_to_pixel(d["point"][1]),
                0,
            ]
        else:
            d["point"] = [
                mm_to_pixel(d["point"][0]),
                mm_to_pixel(d["point"][1]),
                mm_to_pixel(d["point"][2]),
            ]
        points_pixels.append(d)
    data_dict["points"] = points_pixels
    return data_dict


def mm_to_pixel(dist, spacing=SPACING_LEVEL0):
    spacing = spacing / 1000
    dist_px = int(round(dist / spacing))
    return dist_px


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))
