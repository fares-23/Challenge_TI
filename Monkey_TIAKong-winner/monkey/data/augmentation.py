import albumentations as alb
import cv2
import numpy as np


def get_augmentation(
    module: str, gt_type=None, augment=True, aug_prob=0.9
):
    if module == "detection" or module == "multiclass_detection":
        aug = alb.Compose(
            [
                alb.OneOf(
                    [
                        alb.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=0,
                            val_shift_limit=0,
                            always_apply=False,
                            p=0.8,
                        ),  # .8
                        alb.RGBShift(
                            r_shift_limit=20,
                            g_shift_limit=20,
                            b_shift_limit=20,
                            p=0.8,
                        ),  # .7
                    ],
                    p=1,
                ),
                alb.OneOf(
                    [
                        alb.HueSaturationValue(
                            hue_shift_limit=0,
                            sat_shift_limit=(-30, -10),
                            val_shift_limit=0,
                            always_apply=False,
                            p=0.8,
                        ),
                        alb.HueSaturationValue(
                            hue_shift_limit=0,
                            sat_shift_limit=(10, 20),
                            val_shift_limit=0,
                            always_apply=False,
                            p=0.5,
                        ),
                        alb.HueSaturationValue(
                            hue_shift_limit=0,
                            sat_shift_limit=10,
                            val_shift_limit=0,
                            always_apply=False,
                            p=0.8,
                        ),
                    ],
                    p=1,
                ),
                alb.OneOf(
                    [
                        alb.GaussianBlur(blur_limit=(1, 3), p=0.5),
                        alb.Sharpen(
                            alpha=(0.1, 0.3),
                            lightness=(1.0, 1.0),
                            p=0.5,
                        ),
                        alb.ImageCompression(
                            quality_lower=30, quality_upper=80, p=0.5
                        ),
                    ],
                    p=0.5,
                ),
                alb.RandomBrightnessContrast(
                    brightness_limit=0.05, contrast_limit=0.3, p=0.5
                ),
                alb.ShiftScaleRotate(
                    shift_limit=0.01,
                    scale_limit=0.01,
                    rotate_limit=180,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.8,
                ),
                alb.Flip(p=0.8),
            ],
            p=aug_prob,
        )
    else:
        raise ValueError(f"Invalid module {module}")

    return aug
