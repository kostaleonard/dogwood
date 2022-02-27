"""Tests weight_transfer.py."""
# TODO mark slowtests

from typing import Tuple
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from dogwood.errors import NotADenseLayerError
from dogwood.weight_transfer import expand_dense_layer, expand_dense_layers, \
    STRATEGY_ALL_ZERO, STRATEGY_OUTPUT_ZERO, STRATEGY_ALL_RANDOM

MAX_PIXEL_VALUE = 255
MNIST_IMAGE_SHAPE = (28, 28)


@pytest.fixture(scope='session')
def mnist_dataset() -> Tuple[Tuple[np.ndarray, np.ndarray],
                             Tuple[np.ndarray, np.ndarray]]:
    """Returns the preprocessed MNIST dataset.

    :return: The preprocessed MNIST dataset as (X_train, y_train), (X_test,
        y_test).
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = tf.cast(X_train, tf.float32) / MAX_PIXEL_VALUE
    X_test = tf.cast(X_test, tf.float32) / MAX_PIXEL_VALUE
    return (X_train, y_train), (X_test, y_test)


@pytest.fixture
def baseline_model() -> Sequential:
    """Returns the baseline model for use on MNIST.

    :return: The baseline model for use on MNIST.
    """
    model = Sequential([
        Flatten(input_shape=MNIST_IMAGE_SHAPE, name='flatten'),
        Dense(1, activation='relu', name='dense_1'),
        Dense(10, activation='softmax', name='dense_2')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


def test_are_symmetric_dense_neurons_one_neuron() -> None:
    """Tests that one neuron is always considered symmetric."""
    # TODO
    assert False


def test_are_symmetric_dense_neurons_symmetric() -> None:
    """Tests that deliberately symmetric neurons are considered symmetric."""
    # TODO
    assert False


def test_are_symmetric_dense_neurons_constant_initialized() -> None:
    """Tests that a model whose weights were initialized to 0 causes weight
    symmetry."""
    # TODO
    assert False


def test_are_symmetric_dense_neurons_asymmetric() -> None:
    """Tests that a model whose weights were randomly initialized causes no
    weights symmetry."""
    # TODO
    assert False


def test_are_symmetric_dense_neurons_raises_error() -> None:
    """Tests that are_symmetric_dense_neurons raises a NotADenseLayerError when
    supplied with a non-dense layer."""
    # TODO
    assert False


def test_expand_dense_layer_increases_layer_size(
        baseline_model: Sequential) -> None:
    """Tests that expand_dense_layer increases the size of the given layer.

    :param baseline_model: The baseline model.
    """
    num_new_neurons = 5
    expanded = expand_dense_layer(baseline_model, 'dense_1', num_new_neurons)
    assert expanded.get_layer('dense_1').units == \
        baseline_model.get_layer('dense_1').units + num_new_neurons


def test_expand_dense_layer_all_zero_same_output(
        baseline_model: Sequential) -> None:
    """Tests that the all zero strategy does not change model output.

    :param baseline_model: The baseline model.
    """
    num_new_neurons = 5
    expanded = expand_dense_layer(
        baseline_model, 'dense_1', num_new_neurons, strategy=STRATEGY_ALL_ZERO)
    batch = np.ones((2, *MNIST_IMAGE_SHAPE))
    assert np.isclose(baseline_model(batch), expanded(batch)).all()


def test_expand_dense_layer_all_zero_causes_weight_symmetry(
        baseline_model: Sequential,
        mnist_dataset: Tuple[Tuple[np.ndarray, np.ndarray],
                             Tuple[np.ndarray, np.ndarray]]) -> None:
    """Tests that the all zero strategy causes weight symmetry when the model
    is fine-tuned.

    :param baseline_model: The baseline model.
    :param mnist_dataset: The MNIST dataset.
    """
    (X_train, y_train), _ = mnist_dataset
    num_new_neurons = 5
    expanded = expand_dense_layer(
        baseline_model, 'dense_1', num_new_neurons, strategy=STRATEGY_ALL_ZERO)
    expanded.fit(X_train, y_train, epochs=5)
    print(expanded.trainable_weights[0].shape)
    # TODO


def test_expand_dense_layer_output_zero_same_output(
        baseline_model: Sequential) -> None:
    """Tests that the output zero strategy does not change model output.

    :param baseline_model: The baseline model.
    """
    num_new_neurons = 5
    expanded = expand_dense_layer(
        baseline_model,
        'dense_1',
        num_new_neurons,
        strategy=STRATEGY_OUTPUT_ZERO)
    batch = np.ones((2, *MNIST_IMAGE_SHAPE))
    assert np.isclose(baseline_model(batch), expanded(batch)).all()


def test_expand_dense_layer_output_zero_same_output_trained_model(
        baseline_model: Sequential,
        mnist_dataset: Tuple[Tuple[np.ndarray, np.ndarray],
                             Tuple[np.ndarray, np.ndarray]]) -> None:
    """Tests that the output zero strategy does not change model output, even
    when the model has been pretrained.

    :param baseline_model: The baseline model.
    :param mnist_dataset: The MNIST dataset.
    """
    (X_train, y_train), (X_test, y_test) = mnist_dataset
    baseline_model.fit(X_train, y_train, epochs=5)
    baseline_eval = baseline_model.evaluate(X_test, y_test)
    num_new_neurons = 5
    expanded = expand_dense_layer(
        baseline_model,
        'dense_1',
        num_new_neurons,
        strategy=STRATEGY_OUTPUT_ZERO)
    expanded_eval = expanded.evaluate(X_test, y_test)
    assert np.isclose(baseline_eval, expanded_eval).all()


def test_expand_dense_layer_output_zero_no_weight_symmetry() -> None:
    """Tests that the output zero strategy does not cause weight symmetry when
    the model is fine-tuned."""
    # TODO
    assert False


def test_expand_dense_layer_all_random_different_output(
        baseline_model: Sequential) -> None:
    """Tests that the all random strategy changes model output.

    :param baseline_model: The baseline model.
    """
    num_new_neurons = 5
    expanded = expand_dense_layer(
        baseline_model,
        'dense_1',
        num_new_neurons,
        strategy=STRATEGY_ALL_RANDOM)
    batch = np.ones((2, *MNIST_IMAGE_SHAPE))
    assert not np.isclose(baseline_model(batch), expanded(batch)).all()


def test_expand_dense_layer_not_dense_raises_error(
        baseline_model: Sequential) -> None:
    """Tests that expand_dense_layer raises an error when the layer is not
    Dense.

    :param baseline_model: The baseline model.
    """
    with pytest.raises(NotADenseLayerError):
        _ = expand_dense_layer(baseline_model, 'flatten', 1)


def test_expand_dense_layers_increases_layer_sizes() -> None:
    """Tests that expand_dense_layers increases the sizes of the given
    layers.
    """
    # TODO make fixture?
    model = Sequential([
        Flatten(input_shape=MNIST_IMAGE_SHAPE, name='flatten'),
        Dense(1, activation='relu', name='dense_1'),
        Dense(2, activation='relu', name='dense_2'),
        Dense(10, activation='softmax', name='dense_3')
    ])
    dense_layer_names_and_neurons = {'dense_1': 3, 'dense_2': 2}
    expanded = expand_dense_layers(model, dense_layer_names_and_neurons)
    for name, num_new_neurons in dense_layer_names_and_neurons.items():
        assert expanded.get_layer(name).units == \
           model.get_layer(name).units + num_new_neurons


def test_expand_dense_layers_maximizes_number_of_random_weights() -> None:
    """Tests that expand_dense_layers randomly initializes the maximum number
    of weights when strategy is output zero."""
    # TODO
    assert False


def test_expand_dense_layers_all_zero_causes_weight_symmetry() -> None:
    """Tests that expand_dense_layers causes weight symmetry when using the all
    zero strategy."""
    # TODO
    assert False


def test_expand_dense_layers_output_zero_no_weight_symmetry() -> None:
    """Tests that expand_dense_layers does not cause weight symmetry when using
    the output zero strategy."""
    # TODO
    assert False
