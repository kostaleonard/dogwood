"""Tests pretraining_pool.py."""

import os
import shutil
import pytest
import numpy as np
from tensorflow.keras.models import Sequential
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.versioned_model import VersionedModel
from dogwood.pretraining.pretraining_pool import PretrainingPool, \
    MODEL_VGG16, VGG16_VERSION, DATASET_MINI_IMAGENET, MINI_IMAGENET_VERSION
from dogwood.errors import PretrainingPoolAlreadyContainsModelError, \
    NoSuchOpenSourceModelError

TEST_DIRNAME = '/tmp/test_pretraining_pool/pretrained'


def _clear_test_directory() -> None:
    """Clears the test directory."""
    try:
        shutil.rmtree(TEST_DIRNAME)
    except FileNotFoundError:
        pass


def test_init_creates_directory() -> None:
    """Tests that __init__ creates the pretraining directory."""
    _clear_test_directory()
    assert not os.path.exists(TEST_DIRNAME)
    _ = PretrainingPool(TEST_DIRNAME, with_models=None)
    assert os.path.exists(TEST_DIRNAME) and os.path.isdir(TEST_DIRNAME)


@pytest.mark.slowtest
def test_init_gets_models_and_datasets() -> None:
    """Tests that __init__ populates the pretraining directory."""
    _clear_test_directory()
    _ = PretrainingPool(TEST_DIRNAME, with_models=MODEL_VGG16)
    assert os.path.exists(os.path.join(
        TEST_DIRNAME,
        'datasets',
        DATASET_MINI_IMAGENET,
        MINI_IMAGENET_VERSION,
        'X_train.npy'))
    assert os.path.exists(os.path.join(
        TEST_DIRNAME,
        'datasets',
        DATASET_MINI_IMAGENET,
        MINI_IMAGENET_VERSION,
        'y_train.npy'))
    assert os.path.exists(os.path.join(
        TEST_DIRNAME, 'models', MODEL_VGG16, VGG16_VERSION, 'model.h5'))


def test_init_unknown_model_raises_error() -> None:
    """Tests that __init__ raises an error when called with an unknown open
    source model."""
    _clear_test_directory()
    with pytest.raises(NoSuchOpenSourceModelError):
        _ = PretrainingPool(TEST_DIRNAME, with_models='dne')


def test_add_model_writes_versioned_files(
        mnist_model: Sequential,
        mnist_dataset: tuple[tuple[np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray]]) -> None:
    """Tests that add_model writes model and dataset files.

    :param mnist_model: The baseline model.
    :param mnist_dataset: The MNIST dataset.
    """
    _clear_test_directory()
    (X_train, y_train), _ = mnist_dataset
    pool = PretrainingPool(TEST_DIRNAME, with_models=None)
    pool.add_model(mnist_model, X_train, y_train, dataset_name='mnist')
    assert os.path.exists(os.path.join(
        pool.models_dirname, mnist_model.name, 'v1', 'model.h5'))
    assert os.path.exists(os.path.join(
        pool.datasets_dirname, 'mnist', 'v1', 'X_train.npy'))
    assert os.path.exists(os.path.join(
        pool.datasets_dirname, 'mnist', 'v1', 'y_train.npy'))


def test_add_model_reuses_dataset(
        mnist_model: Sequential,
        micro_mnist_model: Sequential,
        mnist_dataset: tuple[tuple[np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray]]) -> None:
    """Tests that add_model allows dataset reuse.

    :param mnist_model: The baseline model.
    :param micro_mnist_model: A small MNIST model.
    :param mnist_dataset: The MNIST dataset.
    """
    _clear_test_directory()
    (X_train, y_train), _ = mnist_dataset
    pool = PretrainingPool(TEST_DIRNAME, with_models=None)
    pool.add_model(mnist_model, X_train, y_train, dataset_name='mnist')
    pool.add_model(micro_mnist_model, X_train, y_train, dataset_name='mnist')
    assert os.path.exists(os.path.join(
        pool.models_dirname, mnist_model.name, 'v1', 'model.h5'))
    assert os.path.exists(os.path.join(
        pool.models_dirname, micro_mnist_model.name, 'v1', 'model.h5'))
    assert os.path.exists(os.path.join(
        pool.datasets_dirname, 'mnist', 'v1', 'X_train.npy'))
    assert os.path.exists(os.path.join(
        pool.datasets_dirname, 'mnist', 'v1', 'y_train.npy'))


def test_add_model_duplicate_model_raises_error(
        mnist_model: Sequential,
        mnist_dataset: tuple[tuple[np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray]]) -> None:
    """Tests that add_model raises an error on a duplicate model.

    :param mnist_model: The baseline model.
    :param mnist_dataset: The MNIST dataset.
    """
    _clear_test_directory()
    (X_train, y_train), _ = mnist_dataset
    pool = PretrainingPool(TEST_DIRNAME, with_models=None)
    pool.add_model(mnist_model, X_train, y_train, dataset_name='mnist')
    with pytest.raises(PretrainingPoolAlreadyContainsModelError):
        pool.add_model(mnist_model, X_train, y_train, dataset_name='mnist')


@pytest.mark.slowtest
def test_add_versioned_model_writes_versioned_files(
        mnist_versioned_dataset: VersionedDataset,
        mnist_versioned_model: VersionedModel) -> None:
    """Tests that add_versioned_model writes model and dataset files.

    :param mnist_versioned_dataset: The versioned MNIST dataset.
    :param mnist_versioned_model: The versioned MNIST model.
    """
    _clear_test_directory()
    pool = PretrainingPool(TEST_DIRNAME, with_models=None)
    pool.add_versioned_model(mnist_versioned_model, mnist_versioned_dataset)
    assert os.path.exists(os.path.join(
        pool.models_dirname, mnist_versioned_model.name, 'v1', 'model.h5'))
    assert os.path.exists(os.path.join(
        pool.datasets_dirname,
        mnist_versioned_dataset.name,
        'v1',
        'X_train.npy'))
    assert os.path.exists(os.path.join(
        pool.datasets_dirname,
        mnist_versioned_dataset.name,
        'v1',
        'y_train.npy'))


@pytest.mark.slowtest
def test_add_versioned_model_reuses_dataset(
        mnist_versioned_dataset: VersionedDataset,
        mnist_versioned_model: VersionedModel) -> None:
    """Tests that add_versioned_model allows dataset reuse.

    :param mnist_versioned_dataset: The versioned MNIST dataset.
    :param mnist_versioned_model: The versioned MNIST model.
    """
    _clear_test_directory()
    pool = PretrainingPool(TEST_DIRNAME, with_models=None)
    original_model_name = mnist_versioned_model.name
    pool.add_versioned_model(mnist_versioned_model, mnist_versioned_dataset)
    mnist_versioned_model.name = 'mnist2'
    pool.add_versioned_model(mnist_versioned_model, mnist_versioned_dataset)
    assert os.path.exists(os.path.join(
        pool.models_dirname, original_model_name, 'v1', 'model.h5'))
    assert os.path.exists(os.path.join(
        pool.models_dirname, mnist_versioned_model.name, 'v1', 'model.h5'))
    assert os.path.exists(os.path.join(
        pool.datasets_dirname,
        mnist_versioned_dataset.name,
        'v1',
        'X_train.npy'))
    assert os.path.exists(os.path.join(
        pool.datasets_dirname,
        mnist_versioned_dataset.name,
        'v1',
        'y_train.npy'))


@pytest.mark.slowtest
def test_add_versioned_model_duplicate_model_raises_error(
        mnist_versioned_dataset: VersionedDataset,
        mnist_versioned_model: VersionedModel) -> None:
    """Tests that add_versioned_model writes model and dataset files.

    :param mnist_versioned_dataset: The versioned MNIST dataset.
    :param mnist_versioned_model: The versioned MNIST model.
    """
    _clear_test_directory()
    pool = PretrainingPool(TEST_DIRNAME, with_models=None)
    pool.add_versioned_model(mnist_versioned_model, mnist_versioned_dataset)
    with pytest.raises(PretrainingPoolAlreadyContainsModelError):
        pool.add_versioned_model(mnist_versioned_model,
                                 mnist_versioned_dataset)


@pytest.mark.slowtest
def test_get_pretrained_model_changes_weights(
        mnist_model: Sequential,
        large_mnist_model: Sequential,
        mnist_dataset: tuple[tuple[np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray]]) -> None:
    """Tests that get_pretrained_model changes model weights.

    :param mnist_model: The baseline model.
    :param large_mnist_model: The large model.
    :param mnist_dataset: The MNIST dataset.
    """
    _clear_test_directory()
    (X_train, y_train), _ = mnist_dataset
    mnist_model.fit(X_train, y_train, batch_size=32, epochs=10)
    pool = PretrainingPool(TEST_DIRNAME, with_models=None)
    pool.add_model(mnist_model, X_train, y_train)
    weights_before = large_mnist_model.get_weights()
    large_mnist_model = pool.get_pretrained_model(
        large_mnist_model, X_train, y_train)
    weights_after = large_mnist_model.get_weights()
    all_equal = True
    for idx, layer_weights_before in enumerate(weights_before):
        layer_weights_after = weights_after[idx]
        if not np.isclose(layer_weights_before, layer_weights_after).all():
            all_equal = False
    assert not all_equal


@pytest.mark.slowtest
def test_get_pretrained_model_improves_performance(
        mnist_model: Sequential,
        large_mnist_model: Sequential,
        mnist_dataset: tuple[tuple[np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray]]) -> None:
    """Tests that get_pretrained_model improves the performance of a new
    model.

    :param mnist_model: The baseline model.
    :param large_mnist_model: The large model.
    :param mnist_dataset: The MNIST dataset.
    """
    _clear_test_directory()
    (X_train, y_train), (X_test, y_test) = mnist_dataset
    mnist_model.fit(X_train, y_train, batch_size=32, epochs=10)
    pool = PretrainingPool(TEST_DIRNAME, with_models=None)
    pool.add_model(mnist_model, X_train, y_train)
    acc_before_transfer = large_mnist_model.evaluate(X_test, y_test)[1]
    large_mnist_model = pool.get_pretrained_model(
        large_mnist_model, X_train, y_train)
    acc_after_transfer = large_mnist_model.evaluate(X_test, y_test)[1]
    assert acc_after_transfer > acc_before_transfer
