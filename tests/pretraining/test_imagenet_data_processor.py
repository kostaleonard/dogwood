"""Tests imagenet_data_processor.py."""

import pytest
import numpy as np
from dogwood.pretraining.imagenet_data_processor import ImageNetDataProcessor

IMAGENET_CLASSES = 1000
CLASS_INDEX_0 = ['n01440764', 'tench']
CLASS_INDEX_980 = ['n09472597', 'volcano']


@pytest.fixture(name='raw_features_and_labels', scope='module')
def fixture_raw_features_and_labels() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """TODO"""
    # TODO


def test_init_gets_class_index() -> None:
    """Tests that __init__ creates the class index."""
    processor = ImageNetDataProcessor()
    assert processor.class_index
    assert set(processor.class_index.keys()) == {
        str(idx) for idx in range(IMAGENET_CLASSES)}
    assert processor.class_index['0'] == CLASS_INDEX_0
    assert processor.class_index['980'] == CLASS_INDEX_980


def test_get_raw_features_and_labels_has_expected_keys() -> None:
    """Tests that"""
