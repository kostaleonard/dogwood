"""Tests weight_transfer.py."""

import pytest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from dogwood.errors import NotADenseLayerError
from dogwood.weight_transfer import expand_dense_layer, expand_dense_layers, \
    STRATEGY_ALL_ZERO, STRATEGY_OUTPUT_ZERO, STRATEGY_ALL_RANDOM

MNIST_IMAGE_SHAPE = (28, 28)


def test_expand_dense_layer_increases_layer_size() -> None:
    """Tests that expand_dense_layer increases the size of the given layer."""
    model = Sequential([
        Flatten(input_shape=MNIST_IMAGE_SHAPE, name='flatten'),
        Dense(1, activation='relu', name='dense_1'),
        Dense(10, activation='softmax', name='dense_2')
    ])
    num_new_neurons = 5
    expanded = expand_dense_layer(model, 'dense_1', num_new_neurons)
    assert expanded.get_layer('dense_1').units == \
        model.get_layer('dense_1').units + num_new_neurons


def test_expand_dense_layer_all_zero_same_output() -> None:
    """Tests that the all zero strategy does not change model output."""
    model = Sequential([
        Flatten(input_shape=MNIST_IMAGE_SHAPE, name='flatten'),
        Dense(1, activation='relu', name='dense_1'),
        Dense(10, activation='softmax', name='dense_2')
    ])
    num_new_neurons = 5
    expanded = expand_dense_layer(
        model, 'dense_1', num_new_neurons, strategy=STRATEGY_ALL_ZERO)
    batch = np.ones((2, *MNIST_IMAGE_SHAPE))
    assert np.isclose(model(batch), expanded(batch)).all()


def test_expand_dense_layer_output_zero_same_output() -> None:
    """Tests that the output zero strategy does not change model output."""
    model = Sequential([
        Flatten(input_shape=MNIST_IMAGE_SHAPE, name='flatten'),
        Dense(1, activation='relu', name='dense_1'),
        Dense(10, activation='softmax', name='dense_2')
    ])
    num_new_neurons = 5
    expanded = expand_dense_layer(
        model, 'dense_1', num_new_neurons, strategy=STRATEGY_OUTPUT_ZERO)
    batch = np.ones((2, *MNIST_IMAGE_SHAPE))
    assert np.isclose(model(batch), expanded(batch)).all()


def test_expand_dense_layer_output_zero_same_output_trained_model() -> None:
    """Tests that the output zero strategy does not change model output, even
    when the model has been pretrained."""
    # TODO
    assert False


def test_expand_dense_layer_all_random_different_output() -> None:
    """Tests that the all random strategy changes model output."""
    model = Sequential([
        Flatten(input_shape=MNIST_IMAGE_SHAPE, name='flatten'),
        Dense(1, activation='relu', name='dense_1'),
        Dense(10, activation='softmax', name='dense_2')
    ])
    num_new_neurons = 5
    expanded = expand_dense_layer(
        model, 'dense_1', num_new_neurons, strategy=STRATEGY_ALL_RANDOM)
    batch = np.ones((2, *MNIST_IMAGE_SHAPE))
    assert not np.isclose(model(batch), expanded(batch)).all()


def test_expand_dense_layer_not_dense_raises_error() -> None:
    """Tests that expand_dense_layer raises an error when the layer is not
    Dense."""
    model = Sequential([
        Flatten(input_shape=MNIST_IMAGE_SHAPE, name='flatten'),
        Dense(1, activation='relu', name='dense_1'),
        Dense(10, activation='softmax', name='dense_2')
    ])
    with pytest.raises(NotADenseLayerError):
        _ = expand_dense_layer(model, 'flatten', 1)


def test_expand_dense_layers_increases_layer_sizes() -> None:
    """Tests that expand_dense_layers increases the sizes of the given
    layers."""
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
