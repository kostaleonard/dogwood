"""Tests pretraining_pool.py."""

import os
import shutil
import pytest
import numpy as np
from tensorflow.keras.models import Sequential
from dogwood.pretraining.pretraining_pool import PretrainingPool

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


def test_init_gets_models_and_datasets() -> None:
    """Tests that __init__ populates the pretraining directory."""
    # TODO likely a slowtest
    _clear_test_directory()
    model_name = 'VGG16'
    _ = PretrainingPool(TEST_DIRNAME, with_models=model_name)
    assert os.path.exists(os.path.join(
        TEST_DIRNAME, model_name, f'{model_name}.h5'))
    assert os.path.exists(os.path.join(
        TEST_DIRNAME, model_name, 'X_train.npy'))
    assert os.path.exists(os.path.join(
        TEST_DIRNAME, model_name, 'y_train.npy'))


def test_add_model_writes_files(
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
    pool.add_model(mnist_model, X_train, y_train)
    assert os.path.exists(os.path.join(
        TEST_DIRNAME, mnist_model.name, f'{mnist_model.name}.h5'))
    assert os.path.exists(os.path.join(
        TEST_DIRNAME, mnist_model.name, 'X_train.npy'))
    assert os.path.exists(os.path.join(
        TEST_DIRNAME, mnist_model.name, 'y_train.npy'))


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
