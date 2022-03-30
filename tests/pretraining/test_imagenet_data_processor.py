"""Tests imagenet_data_processor.py."""
# pylint: disable=no-name-in-module

import os
import pytest
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from dogwood.pretraining.mini_imagenet_loader import download_mini_imagenet
from dogwood.pretraining.imagenet_data_processor import \
    ImageNetDataProcessor, IMAGE_SCALE_HEIGHT, IMAGE_SCALE_WIDTH, \
    IMAGE_TARGET_SIZE

IMAGENET_CLASSES = 1000
CLASS_INDEX_0 = ['n01440764', 'tench']
CLASS_INDEX_980 = ['n09472597', 'volcano']
TEST_ROOT_PATH = '/tmp/test_imagenet_data_processor/'
TEST_IMAGENET_PATH = os.path.join(TEST_ROOT_PATH, 'imagenet-mini')
CLASS_INDEX_0_TRAIN_PATH = os.path.join(
    TEST_IMAGENET_PATH, 'train', CLASS_INDEX_0[0])
CLASS_INDEX_0_IMG_PATH = os.path.join(
    CLASS_INDEX_0_TRAIN_PATH, 'n01440764_10043.JPEG')
NUM_CHANNELS = 3
PIXEL_MIN = 0
PIXEL_MAX = 255


@pytest.fixture(name='raw_features_and_labels', scope='module')
def fixture_raw_features_and_labels() -> \
        tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Returns the raw features and labels.

    :return: The raw features and labels.
    """
    if not os.path.exists(TEST_IMAGENET_PATH):
        download_mini_imagenet(TEST_ROOT_PATH)
    processor = ImageNetDataProcessor()
    return processor.get_raw_features_and_labels(TEST_IMAGENET_PATH)


@pytest.fixture(name='raw_features', scope='module')
def fixture_raw_features() -> dict[str, np.ndarray]:
    """Returns the raw features.

    :return: The raw features.
    """
    if not os.path.exists(TEST_IMAGENET_PATH):
        download_mini_imagenet(TEST_ROOT_PATH)
    processor = ImageNetDataProcessor()
    return processor.get_raw_features(TEST_IMAGENET_PATH)


def test_init_gets_class_index() -> None:
    """Tests that __init__ creates the class index."""
    processor = ImageNetDataProcessor()
    assert processor.class_index
    assert set(processor.class_index.keys()) == {
        str(idx) for idx in range(IMAGENET_CLASSES)}
    assert processor.class_index['0'] == CLASS_INDEX_0
    assert processor.class_index['980'] == CLASS_INDEX_980


def test_init_gets_designator_index() -> None:
    """Tests that __init__ creates the designator index."""
    processor = ImageNetDataProcessor()
    assert processor.designator_index
    assert len(processor.designator_index) == len(processor.class_index)
    assert CLASS_INDEX_0[0] in processor.designator_index.keys()
    assert processor.designator_index[CLASS_INDEX_0[0]] == '0'
    assert CLASS_INDEX_980[0] in processor.designator_index.keys()
    assert processor.designator_index[CLASS_INDEX_980[0]] == '980'


@pytest.mark.slowtest
def test_get_raw_features_and_labels_has_expected_keys(
        raw_features_and_labels: tuple[dict[str, np.ndarray],
                                       dict[str, np.ndarray]]) -> None:
    """Tests that the raw features and labels have the expected keys.

    :param raw_features_and_labels: The raw features and labels.
    """
    raw_features, raw_labels = raw_features_and_labels
    assert set(raw_features.keys()) == {'X_train', 'X_val'}
    assert set(raw_labels.keys()) == {'y_train', 'y_val'}


@pytest.mark.slowtest
def test_get_raw_features_and_labels_has_all_classes(
        raw_features_and_labels: tuple[dict[str, np.ndarray],
                                       dict[str, np.ndarray]]) -> None:
    """Tests that the raw features and labels have all classes.

    :param raw_features_and_labels: The raw features and labels.
    """
    _, raw_labels = raw_features_and_labels
    assert len(set(np.unique(raw_labels['y_train']))) == IMAGENET_CLASSES
    assert len(set(np.unique(raw_labels['y_val']))) == IMAGENET_CLASSES


@pytest.mark.slowtest
def test_get_raw_features_and_labels_correct_shape(
        raw_features_and_labels: tuple[dict[str, np.ndarray],
                                       dict[str, np.ndarray]]) -> None:
    """Tests that the raw features and labels are the correct shape.

    :param raw_features_and_labels: The raw features and labels.
    """
    raw_features, raw_labels = raw_features_and_labels
    assert raw_features['X_train'].shape[1:] == (
        IMAGE_SCALE_HEIGHT, IMAGE_SCALE_WIDTH, NUM_CHANNELS)
    assert raw_features['X_val'].shape[1:] == (
        IMAGE_SCALE_HEIGHT, IMAGE_SCALE_WIDTH, NUM_CHANNELS)
    assert len(raw_labels['y_train'].shape) == 1
    assert len(raw_labels['y_val'].shape) == 1
    assert len(raw_features['X_train']) == len(raw_labels['y_train'])
    assert len(raw_features['X_val']) == len(raw_labels['y_val'])


@pytest.mark.slowtest
def test_get_raw_features_and_labels_correct_dtype(
        raw_features_and_labels: tuple[dict[str, np.ndarray],
                                       dict[str, np.ndarray]]) -> None:
    """Tests that the raw features and labels are the correct data type.

    :param raw_features_and_labels: The raw features and labels.
    """
    raw_features, raw_labels = raw_features_and_labels
    assert raw_features['X_train'].dtype == np.uint8
    assert raw_features['X_val'].dtype == np.uint8
    assert raw_features['X_train'].max() == PIXEL_MAX
    assert raw_features['X_train'].min() == PIXEL_MIN
    assert raw_features['X_val'].max() == PIXEL_MAX
    assert raw_features['X_val'].min() == PIXEL_MIN
    assert isinstance(raw_labels['y_train'][0], str)
    assert isinstance(raw_labels['y_val'][0], str)


@pytest.mark.slowtest
def test_get_raw_features_and_labels_valid_labels(
        raw_features_and_labels: tuple[dict[str, np.ndarray],
                                       dict[str, np.ndarray]]) -> None:
    """Tests that the raw features and labels have valid labels.

    :param raw_features_and_labels: The raw features and labels.
    """
    _, raw_labels = raw_features_and_labels
    class_index = ImageNetDataProcessor.get_class_index()
    class_designators = {designator_and_name[0]
                         for designator_and_name in class_index.values()}
    for subset in 'y_train', 'y_val':
        for label in raw_labels[subset]:
            assert label in class_designators


@pytest.mark.slowtest
def test_get_raw_features_and_labels_match(
        raw_features_and_labels: tuple[dict[str, np.ndarray],
                                       dict[str, np.ndarray]]) -> None:
    """Tests that the raw features and labels are correctly aligned.

    :param raw_features_and_labels: The raw features and labels.
    """
    raw_features, raw_labels = raw_features_and_labels
    num_class_0 = len(os.listdir(CLASS_INDEX_0_TRAIN_PATH))
    class_0_indices = raw_labels['y_train'] == CLASS_INDEX_0[0]
    assert len(raw_labels['y_train'][class_0_indices]) == num_class_0
    image = load_img(CLASS_INDEX_0_IMG_PATH, target_size=IMAGE_TARGET_SIZE)
    target_tensor = img_to_array(image, dtype=np.uint8)
    assert target_tensor.shape == raw_features['X_train'].shape[1:]
    found_image = False
    for train_tensor in raw_features['X_train'][class_0_indices]:
        if np.array_equal(train_tensor, target_tensor):
            found_image = True
    assert found_image


@pytest.mark.slowtest
def test_get_raw_features_gets_train_and_val(
        raw_features: dict[str, np.ndarray],
        raw_features_and_labels: tuple[dict[str, np.ndarray],
                                       dict[str, np.ndarray]]) -> None:
    """Tests that get_raw_features gets the train and val sets.

    :param raw_features: The raw features.
    :param raw_features_and_labels: The raw features and labels.
    """
    raw_features_from_labels, _ = raw_features_and_labels
    assert set(raw_features.keys()) == set(raw_features_from_labels.keys())
    for subset, raw_tensor in raw_features.items():
        raw_tensor_from_labels = raw_features_from_labels[subset]
        assert raw_tensor.shape == raw_tensor_from_labels.shape
        # Order may be different.
        assert np.array_equal(raw_tensor.flatten().sort(),
                              raw_tensor_from_labels.flatten().sort())


def test_preprocess_features_is_identity_function() -> None:
    """Tests that preprocess_features is the identity function."""
    processor = ImageNetDataProcessor()
    arr = np.array([[1, 2], [3, 4]])
    assert np.array_equal(arr, processor.preprocess_features(arr))


def test_unpreprocess_features_is_identity_function() -> None:
    """Tests that unpreprocess_features is the identity function."""
    processor = ImageNetDataProcessor()
    arr = np.array([[1, 2], [3, 4]])
    assert np.array_equal(arr, processor.unpreprocess_features(arr))


def test_preprocess_labels_correct_index() -> None:
    """Tests that preprocess_labels maps names to the correct index."""
    processor = ImageNetDataProcessor()
    raw_labels = np.array([CLASS_INDEX_0[0], CLASS_INDEX_980[0]])
    preprocessed_labels = processor.preprocess_labels(raw_labels)
    assert preprocessed_labels.shape == (len(raw_labels), IMAGENET_CLASSES)
    assert preprocessed_labels[0, 0] == 1
    assert preprocessed_labels[1, 980] == 1


def test_preprocess_labels_is_one_hot() -> None:
    """Tests that preprocess_labels returns a one-hot mapping."""
    processor = ImageNetDataProcessor()
    raw_labels = np.array([CLASS_INDEX_0[0], CLASS_INDEX_980[0]])
    preprocessed_labels = processor.preprocess_labels(raw_labels)
    row_sums = preprocessed_labels.sum(axis=1)
    assert (row_sums == 1).all()
    assert set(np.unique(preprocessed_labels)) == {0, 1}


def test_unpreprocess_labels_inverts_preprocessing() -> None:
    """Tests that unpreprocess_labels inverts preprocess_labels."""
    processor = ImageNetDataProcessor()
    raw_labels = np.array([CLASS_INDEX_0[0], CLASS_INDEX_980[0]])
    preprocessed_labels = processor.preprocess_labels(raw_labels)
    unpreprocessed_labels = processor.unpreprocess_labels(preprocessed_labels)
    assert np.array_equal(raw_labels, unpreprocessed_labels)
