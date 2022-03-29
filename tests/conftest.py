"""Defines pytest fixtures."""

import os
import pytest
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.dataset.pathless_versioned_dataset_builder import \
    PathlessVersionedDatasetBuilder
from mlops.model.versioned_model import VersionedModel
from mlops.model.versioned_model_builder import VersionedModelBuilder
from mlops.model.training_config import TrainingConfig
from mlops.errors import PublicationPathAlreadyExistsError

MAX_PIXEL_VALUE = 255
MNIST_IMAGE_SHAPE = (28, 28)
MICRO_INPUT_LEN = 2
MICRO_HIDDEN_LEN = 4
MICRO_OUTPUT_LEN = 3
FIXTURES_PATH = '/tmp/dogwood/fixtures'
DATASET_FIXTURES_PATH = os.path.join(FIXTURES_PATH, 'datasets')
MODEL_FIXTURES_PATH = os.path.join(FIXTURES_PATH, 'models')
MNIST_DATASET_PUBLICATION_PATH = os.path.join(DATASET_FIXTURES_PATH, 'mnist')
MNIST_MODEL_PUBLICATION_PATH = os.path.join(MODEL_FIXTURES_PATH, 'mnist')


@pytest.fixture(scope='session', name='mnist_dataset')
def fixture_mnist_dataset() -> tuple[tuple[np.ndarray, np.ndarray],
                                     tuple[np.ndarray, np.ndarray]]:
    """Returns the preprocessed MNIST dataset.

    :return: The preprocessed MNIST dataset as (X_train, y_train), (X_test,
        y_test).
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype(np.float32) / MAX_PIXEL_VALUE
    X_test = X_test.astype(np.float32) / MAX_PIXEL_VALUE
    return (X_train, y_train), (X_test, y_test)


@pytest.fixture(name='micro_mnist_model')
def fixture_micro_mnist_model() -> Sequential:
    """Returns the micro model for use on MNIST.

    :return: The micro model for use on MNIST.
    """
    model = Sequential([
        Flatten(input_shape=MNIST_IMAGE_SHAPE, name='flatten'),
        Dense(1, activation='relu', name='dense_1'),
        Dense(10, activation='softmax', name='dense_2')
    ], name='micro_mnist_model')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


@pytest.fixture(name='multilayer_mnist_model')
def fixture_multilayer_mnist_model() -> Sequential:
    """Returns a multilayer model for use on MNIST.

    :return: A multilayer model for use on MNIST.
    """
    model = Sequential([
        Flatten(input_shape=MNIST_IMAGE_SHAPE, name='flatten'),
        Dense(1, activation='relu', name='dense_1'),
        Dense(2, activation='relu', name='dense_2'),
        Dense(10, activation='softmax', name='dense_3')
    ], name='multilayer_mnist_model')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


@pytest.fixture(name='mnist_model')
def fixture_mnist_model() -> Sequential:
    """Returns the baseline model for use on MNIST.

    :return: The baseline model for use on MNIST.
    """
    model = Sequential([
        Flatten(input_shape=MNIST_IMAGE_SHAPE, name='flatten'),
        Dense(128, activation='relu', name='dense_1'),
        Dense(10, activation='softmax', name='dense_2')
    ], name='mnist_model')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


@pytest.fixture(name='large_mnist_model')
def fixture_large_mnist_model() -> Sequential:
    """Returns the large model for use on MNIST.

    :return: The large model for use on MNIST.
    """
    model = Sequential([
        Flatten(input_shape=MNIST_IMAGE_SHAPE, name='flatten'),
        Dense(256, activation='relu', name='dense_1'),
        Dense(10, activation='softmax', name='dense_2')
    ], name='large_mnist_model')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


@pytest.fixture(name='micro_symmetry_model')
def fixture_micro_symmetry_model() -> Sequential:
    """Returns the small model used in weight symmetry tests.

    :return: The small model used in weight symmetry tests.
    """
    model = Sequential([
        Dense(MICRO_HIDDEN_LEN, activation='relu', name='dense_1',
              input_shape=(MICRO_INPUT_LEN,)),
        Dense(MICRO_OUTPUT_LEN, activation='softmax', name='dense_2')
    ])
    model.compile(loss='sparse_categorical_crossentropy')
    return model


@pytest.fixture(name='micro_symmetry_dataset')
def fixture_micro_symmetry_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Returns a training dataset for the micro symmetry model.

    :return: A training dataset for the micro symmetry model as X_train,
        y_train.
    """
    num_examples = 5
    X_train = np.arange(
        num_examples * MICRO_INPUT_LEN, dtype=np.float32).reshape(
        (num_examples, MICRO_INPUT_LEN))
    y_train = np.arange(num_examples, dtype=np.float32) % MICRO_OUTPUT_LEN
    return X_train, y_train


@pytest.fixture(scope='session', name='mnist_versioned_dataset')
def fixture_mnist_versioned_dataset(
        mnist_dataset: tuple[tuple[np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray]]) -> \
        VersionedDataset:
    """Returns the preprocessed MNIST dataset.

    :param mnist_dataset: The MNIST dataset.
    :return: The preprocessed MNIST dataset as a VersionedDataset.
    """
    (X_train, y_train), (X_test, y_test) = mnist_dataset
    features = {'X_train': X_train, 'X_test': X_test}
    labels = {'y_train': y_train, 'y_test': y_test}
    dataset_builder = PathlessVersionedDatasetBuilder(features, labels)
    try:
        dataset_builder.publish(
            MNIST_DATASET_PUBLICATION_PATH,
            name='mnist',
            version='v1',
            tags=['fixture'])
    except PublicationPathAlreadyExistsError:
        pass
    return VersionedDataset(os.path.join(MNIST_DATASET_PUBLICATION_PATH, 'v1'))


@pytest.fixture(name='mnist_versioned_model')
def fixture_mnist_versioned_model(
        mnist_versioned_dataset: VersionedDataset,
        mnist_model: Sequential) -> VersionedModel:
    """Returns the versioned baseline model for use on MNIST.

    :param mnist_versioned_dataset: The MNIST VersionedDataset.
    :param mnist_model: The MNIST baseline model.
    :return: The versioned baseline model for use on MNIST. The model is fit
        to the dataset.
    """
    train_args = {'batch_size': 32, 'epochs': 10}
    history = mnist_model.fit(
        mnist_versioned_dataset.X_train,
        mnist_versioned_dataset.y_train,
        batch_size=train_args['batch_size'],
        epochs=train_args['epochs'])
    training_config = TrainingConfig(history, train_args)
    model_builder = VersionedModelBuilder(
        mnist_versioned_dataset, mnist_model, training_config=training_config)
    try:
        model_builder.publish(
            MNIST_MODEL_PUBLICATION_PATH,
            name='mnist',
            version='v1',
            tags=['fixture'])
    except PublicationPathAlreadyExistsError:
        pass
    return VersionedModel(os.path.join(MNIST_MODEL_PUBLICATION_PATH, 'v1'))
