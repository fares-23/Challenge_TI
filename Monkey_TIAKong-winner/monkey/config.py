import os

# Change this
DEFAULT_DATA_DIR = "/home/u1910100/Documents/Monkey/patches_256"


class TrainingIOConfig:
    def __init__(
        self,
        dataset_dir: str = DEFAULT_DATA_DIR,
        save_dir: str = "./",
    ):
        """IO config for training

        Args:
            dataset_dir (str, optional): Defaults to DEFAULT_DATA_DIR.
            save_dir (str, optional): Defaults to "./".
        """
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.image_dir = ""
        self.mask_dir = ""
        self.json_dir = ""
        self.checkpoint_save_dir = ""

        image_dir = os.path.join(self.dataset_dir, "images/")
        mask_dir = os.path.join(
            self.dataset_dir, "annotations/masks/"
        )
        self.cell_centroid_mask_dir = os.path.join(
            self.dataset_dir, "annotations/masks/"
        )
        json_dir = os.path.join(self.dataset_dir, "annotations/json/")

        self.set_image_dir(image_dir)
        self.set_mask_dir(mask_dir)
        self.set_json_dir(json_dir)

    def set_image_dir(self, image_dir: str):
        """Set patches directory

        Args:
            image_dir (str): path to patches
        """
        self.image_dir = image_dir

    def set_mask_dir(self, mask_dir: str):
        """Set mask directory

        Args:
            mask_dir (str): path to masks
        """
        self.mask_dir = mask_dir

    def set_json_dir(self, json_dir: str):
        """Set json directory

        Args:
            json_dir (str): path to json annotations
        """
        self.json_dir = json_dir

    def set_checkpoint_save_dir(self, run_name: str):
        """Set checkpoint save directory

        Args:
            run_name (str): path to save checkpoints
        """
        dir = os.path.join(self.save_dir, run_name)
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        self.checkpoint_save_dir = dir

    def check_dirs_exist(self):
        """Check of image dir, mask dir and save dir exist.

        Raises:
            ValueError
        """
        for dir in [self.image_dir, self.mask_dir, self.save_dir]:
            if not os.path.exists(dir):
                print(f"{dir} does not exist!")
                raise ValueError(f"{dir} does not exist!")


class PredictionIOConfig:
    def __init__(
        self,
        wsi_dir: str,
        mask_dir: str,
        output_dir: str,
        patch_size: int = 256,
        resolution: float = 0,
        units: str = "level",
        stride: int = 256,
        threshold: float = 0.9,
        thresholds: list = [0.3, 0.3, 0.3],
        min_size: int = 96,
        include_background: bool = False,
        min_distances: list = [5, 5, 5],
        nms_boxes: list = [30, 16, 40],
        nms_overlap_thresh: float = 0.5,
        seg_model_version: int = 1,
    ):
        """Initalize Config

        Args:
            wsi_dir (str): path to wsi dir
            mask_dir (str): path to mask dir
            output_dir (str): path to output dir
            patch_size (int, optional): patch extraction size. Defaults to 256.
            resolution (float, optional): patch extraction resoltuion. Defaults to 0.
            units (str, optional): patch extraction resolution. Defaults to "level".
            stride (int, optional): patch extraction stride. Defaults to 256.
            threshold (float, optional): for binary prediction. Defaults to 0.9.
            thresholds (list, optional): for multiclass prediction. Defaults to [0.3, 0.3, 0.3].
            min_size (int, optional): no used. Defaults to 96.
            include_background (bool, optional): not used. Defaults to False.
            min_distances (list, optional): min distance for peak local max. Defaults to [5, 5, 5].
            nms_boxes (list, optional): box size for nms
            nms_overlap_thresh: overlap thresh for nms
            seg_model_version: version of segmentation model
        """
        self.wsi_dir = wsi_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.check_dirs_exist()
        self.patch_size = patch_size
        self.stride = stride
        self.resolution = resolution
        self.units = units
        self.threshold = threshold
        self.min_size = min_size
        self.include_background = include_background
        self.thresholds = thresholds
        self.min_distances = min_distances
        self.nms_boxes = nms_boxes
        self.nms_overlap_thresh = nms_overlap_thresh
        self.seg_model_version = seg_model_version

    def check_dirs_exist(self):
        """Check of wsi dir, mask dir and output dir exist.
        Raises:
            ValueError: _description_
        """
        for dir in [self.wsi_dir, self.mask_dir, self.output_dir]:
            if not os.path.exists(dir):
                print(f"{dir} does not exist!")
                raise ValueError(f"{dir} does not exist!")
