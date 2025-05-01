import os

DATA_DIR = "/home/Christelle/Documents/CHALLENGE/images"

class SimpleTrainingConfig:
    def __init__(self, dataset_dir= DATA_DIR, save_dir="./results"):
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        # Dossiers standards attendus
        self.image_dir = os.path.join(dataset_dir, "images/")
        self.mask_dir = os.path.join(dataset_dir, "annotations/xml/")


        self.check_dirs_exist()

    def check_dirs_exist(self):
        for d in [self.image_dir, self.mask_dir]:
            if not os.path.exists(d):
                raise FileNotFoundError(f"Le dossier {d} est introuvable.")


class SimplePredictionConfig:
    def __init__(self, image_dir, mask_dir, output_dir="./predictions"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        self.check_dirs_exist()

        # Paramètres par défaut
        self.patch_size = 256
        self.stride = 256
        self.threshold = 0.5

    def check_dirs_exist(self):
        for d in [self.image_dir, self.mask_dir, self.output_dir]:
            if not os.path.exists(d):
                raise FileNotFoundError(f"Le dossier {d} est introuvable.")
