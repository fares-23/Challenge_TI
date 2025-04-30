# Singular pipeline for lymphocyte and monocyte detection
# Detect and classify cells using a single model

import os
from pprint import pprint

import click
import torch
import ttach as tta
from tiatoolbox.wsicore.wsireader import WSIReader
from tqdm.auto import tqdm

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import (
    extract_id,
    open_json_file,
    save_detection_records_monkey,
)
from monkey.model.multihead_model.model import get_multihead_model
from prediction.multiclass_detection import wsi_detection_in_mask_v2


@click.command()
@click.option("--fold", default=1)
def cross_validation(fold: int = 1):
    # detector_model_name = "efficientnetv2_l_multitask_det_decoder_v4_final"
    detector_model_name = "efficientnetv2_l_multitask_det_decoder_v4"
    pprint(f"Multiclass detection using {detector_model_name}")
    pprint(f"Fold {fold}")
    model_res = 0
    units = "level"
    pprint(f"Detect at {model_res} {units}")

    config = PredictionIOConfig(
        wsi_dir="/mnt/lab-share/Monkey/Dataset/images/pas-cpg",
        mask_dir="/mnt/lab-share/Monkey/Dataset/images/tissue-masks",
        output_dir=f"/home/u1910100/cloud_workspace/data/Monkey/local_output/{detector_model_name}_1_2/Fold_{fold}",
        patch_size=256,
        resolution=model_res,
        units=units,
        stride=224,
        thresholds=[0.5, 0.5, 0.5],
        min_distances=[11, 11, 11],
        nms_boxes=[11, 11, 11],
        nms_overlap_thresh=0.5,
    )

    split_info = open_json_file(
        "/mnt/lab-share/Monkey/patches_256/wsi_level_split.json"
    )

    val_wsi_files = split_info[f"Fold_{fold}"]["test_files"]

    print(val_wsi_files)

    # Load models
    detector_weight_paths = [
        f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{detector_model_name}/fold_1/best_val.pth",
        f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{detector_model_name}/fold_2/best_val.pth",
        f"/home/u1910100/cloud_workspace/data/Monkey/cell_multiclass_det/{detector_model_name}/fold_4/best_val.pth",
    ]
    detectors = []
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ]
    )
    for weight_path in detector_weight_paths:
        model = get_multihead_model(
            enc="tf_efficientnetv2_l.in21k_ft_in1k",
            pretrained=True,
            use_batchnorm=True,
            attention_type="scse",
            decoders_out_channels=[3, 3, 3],
            center=True,
        )
        checkpoint = torch.load(weight_path)
        print(f"epoch: {checkpoint['epoch']}")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        model.to("cuda")
        model = tta.SegmentationTTAWrapper(model, transforms)
        detectors.append(model)

    for wsi_name in tqdm(val_wsi_files):
        wsi_name_without_ext = os.path.splitext(wsi_name)[0]
        wsi_id = extract_id(wsi_name)
        mask_name = f"{wsi_id}_mask.tif"

        detection_records = wsi_detection_in_mask_v2(
            wsi_name, mask_name, config, detectors
        )

        inflamm_records = detection_records["inflamm_records"]
        lymph_records = detection_records["lymph_records"]
        mono_records = detection_records["mono_records"]
        print(f"{len(inflamm_records)} final detected inflamm")
        print(f"{len(lymph_records)} final detected lymph")
        print(f"{len(mono_records)} final detected mono")

        wsi_dir = config.wsi_dir
        wsi_path = os.path.join(wsi_dir, wsi_name)
        wsi_reader = WSIReader.open(wsi_path)
        base_mpp = wsi_reader.convert_resolution_units(
            input_res=0, input_unit="level", output_unit="mpp"
        )[0]
        save_detection_records_monkey(
            config,
            inflamm_records,
            lymph_records,
            mono_records,
            wsi_id=wsi_id,
            save_mpp=base_mpp,
        )
        print("finished")


if __name__ == "__main__":
    cross_validation()
