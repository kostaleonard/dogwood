"""Defines pytest fixtures."""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

MAX_PIXEL_VALUE = 255
MNIST_IMAGE_SHAPE = (28, 28)
MICRO_INPUT_LEN = 2
MICRO_HIDDEN_LEN = 4
MICRO_OUTPUT_LEN = 3


@pytest.fixture(scope='session', name='mnist_dataset')
def fixture_mnist_dataset() -> tuple[tuple[np.ndarray, np.ndarray],
                                     tuple[np.ndarray, np.ndarray]]:
    """Returns the preprocessed MNIST dataset.

    :return: The preprocessed MNIST dataset as (X_train, y_train), (X_test,
        y_test).
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = tf.cast(X_train, tf.float32) / MAX_PIXEL_VALUE
    X_test = tf.cast(X_test, tf.float32) / MAX_PIXEL_VALUE
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
