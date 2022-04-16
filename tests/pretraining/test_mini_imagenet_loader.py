"""Tests mini_imagenet_loader.py."""

import os
import shutil
import pytest
from dogwood.pretraining.mini_imagenet_loader import download_mini_imagenet

TEST_IMAGENET_PATH = "/tmp/test_mini_imagenet_loader/mini_imagenet"
IMAGENET_CLASSES = 1000


@pytest.mark.slowtest
@pytest.mark.veryslowtest
def test_download_mini_imagenet_retrieves_files() -> None:
    """Tests that download_mini_imagenet gets the dataset."""
    shutil.rmtree(TEST_IMAGENET_PATH, ignore_errors=True)
    download_mini_imagenet(TEST_IMAGENET_PATH)
    assert os.path.exists(TEST_IMAGENET_PATH)
    assert os.path.exists(os.path.join(TEST_IMAGENET_PATH, "imagenet-mini"))
    assert os.path.exists(
        os.path.join(TEST_IMAGENET_PATH, "imagenet-mini", "train")
    )
    assert (
        len(
            os.listdir(
                os.path.join(TEST_IMAGENET_PATH, "imagenet-mini", "train")
            )
        )
        == IMAGENET_CLASSES
    )
    assert os.path.exists(
        os.path.join(TEST_IMAGENET_PATH, "imagenet-mini", "val")
    )
    assert (
        len(
            os.listdir(
                os.path.join(TEST_IMAGENET_PATH, "imagenet-mini", "val")
            )
        )
        == IMAGENET_CLASSES
    )
    shutil.rmtree(TEST_IMAGENET_PATH, ignore_errors=True)
