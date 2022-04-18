"""Loads Mini-ImageNet from Kaggle."""

import kaggle

MINI_IMAGENET_DATASET = "ifigotin/imagenetmini-1000"
MINI_IMAGENET_DIRNAME = "imagenet-mini"


def download_mini_imagenet(save_path: str) -> None:
    """Downloads Mini-ImageNet from Kaggle and saves it to the given path.

    :param save_path: The directory to which to save the dataset.
    """
    kaggle.api.dataset_download_files(
        MINI_IMAGENET_DATASET, path=save_path, quiet=False, unzip=True
    )
