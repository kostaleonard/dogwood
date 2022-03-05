"""Tests weight_transfer.py."""
# pylint: disable=no-name-in-module
# TODO mark slowtests

import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from dogwood.errors import NotADenseLayerError, InvalidExpansionStrategyError
from dogwood.weight_transfer import expand_dense_layer, expand_dense_layers, \
    STRATEGY_ALL_ZERO, STRATEGY_OUTPUT_ZERO, STRATEGY_ALL_RANDOM, \
    are_symmetric_dense_neurons, clone_layer

MAX_PIXEL_VALUE = 255
MNIST_IMAGE_SHAPE = (28, 28)
MICRO_INPUT_LEN = 2
MICRO_HIDDEN_LEN = 4
MICRO_OUTPUT_LEN = 3


@pytest.fixture(scope='session')
def mnist_dataset() -> tuple[tuple[np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray]]:
    """Returns the preprocessed MNIST dataset.

    :return: The preprocessed MNIST dataset as (X_train, y_train), (X_test,
        y_test).
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = tf.cast(X_train, tf.float32) / MAX_PIXEL_VALUE
    X_test = tf.cast(X_test, tf.float32) / MAX_PIXEL_VALUE
    return (X_train, y_train), (X_test, y_test)


@pytest.fixture
def mnist_model() -> Sequential:
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


@pytest.fixture
def multilayer_mnist_model() -> Sequential:
    """Returns a multilayer model for use on MNIST.

    :return: A multilayer model for use on MNIST.
    """
    model = Sequential([
        Flatten(input_shape=MNIST_IMAGE_SHAPE, name='flatten'),
        Dense(1, activation='relu', name='dense_1'),
        Dense(2, activation='relu', name='dense_2'),
        Dense(10, activation='softmax', name='dense_3')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


@pytest.fixture
def micro_symmetry_model() -> Sequential:
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


@pytest.fixture
def micro_symmetry_dataset() -> tuple[np.ndarray, np.ndarray]:
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


def test_are_symmetric_dense_neurons_one_neuron(
        micro_symmetry_model: Sequential) -> None:
    """Tests that one neuron is always considered symmetric.

    :param micro_symmetry_model: The small model used in weight symmetry tests.
    """
    for layer_name in 'dense_1', 'dense_2':
        for neuron_idx in range(micro_symmetry_model.get_layer(
                layer_name).units):
            assert are_symmetric_dense_neurons(
                micro_symmetry_model, layer_name, {neuron_idx})


def test_are_symmetric_dense_neurons_symmetric(
        micro_symmetry_model: Sequential,
        micro_symmetry_dataset: tuple[np.ndarray, np.ndarray]) -> None:
    """Tests that deliberately symmetric neurons are considered symmetric.

    :param micro_symmetry_model: The small model used in weight symmetry tests.
    :param micro_symmetry_dataset: The training dataset for the model.
    """
    # Deliberately set the weights of neurons 3 and 4 in dense_1.
    weights = micro_symmetry_model.get_weights()
    # Weights in.
    weights[0][:, 2:4] = np.ones((2, 2))
    # Biases in.
    weights[1][2:4] = np.ones((2,))
    # Weights out.
    weights[2][2:4, :] = np.ones((2, 3))
    micro_symmetry_model.set_weights(weights)
    assert are_symmetric_dense_neurons(
        micro_symmetry_model,
        'dense_1',
        {2, 3}
    )
    X_train, y_train = micro_symmetry_dataset
    micro_symmetry_model.fit(X_train, y_train, epochs=5)
    assert are_symmetric_dense_neurons(
        micro_symmetry_model,
        'dense_1',
        {2, 3}
    )


def test_are_symmetric_dense_neurons_asymmetric_weights_in(
        micro_symmetry_model: Sequential) -> None:
    """Tests that a model whose weights into a neuron are different is
    asymmetric.

    :param micro_symmetry_model: The small model used in weight symmetry tests.
    """
    # Deliberately set the weights of neurons 3 and 4 in dense_1.
    weights = micro_symmetry_model.get_weights()
    # Weights in.
    weights[0][:, 2:4] = np.ones((2, 2))
    # Biases in.
    weights[1][2:4] = np.ones((2,))
    # Weights out.
    weights[2][2:4, :] = np.ones((2, 3))
    micro_symmetry_model.set_weights(weights)
    assert are_symmetric_dense_neurons(
        micro_symmetry_model,
        'dense_1',
        {2, 3}
    )
    # Cause asymmetry in weights in.
    weights[0][:, 2:4] = np.array([[1, 2], [1, 2]])
    micro_symmetry_model.set_weights(weights)
    assert not are_symmetric_dense_neurons(
        micro_symmetry_model,
        'dense_1',
        {2, 3}
    )


def test_are_symmetric_dense_neurons_asymmetric_biases_in(
        micro_symmetry_model: Sequential) -> None:
    """Tests that a model whose biases into a neuron are different is
    asymmetric.

    :param micro_symmetry_model: The small model used in weight symmetry tests.
    """
    # Deliberately set the weights of neurons 3 and 4 in dense_1.
    weights = micro_symmetry_model.get_weights()
    # Weights in.
    weights[0][:, 2:4] = np.ones((2, 2))
    # Biases in.
    weights[1][2:4] = np.ones((2,))
    # Weights out.
    weights[2][2:4, :] = np.ones((2, 3))
    micro_symmetry_model.set_weights(weights)
    assert are_symmetric_dense_neurons(
        micro_symmetry_model,
        'dense_1',
        {2, 3}
    )
    # Cause asymmetry in biases in.
    weights[1][2:4] = np.array([1, 2])
    micro_symmetry_model.set_weights(weights)
    assert not are_symmetric_dense_neurons(
        micro_symmetry_model,
        'dense_1',
        {2, 3}
    )


def test_are_symmetric_dense_neurons_asymmetric_weights_out(
        micro_symmetry_model: Sequential) -> None:
    """Tests that a model whose weights out of a neuron are different is
    asymmetric.

    :param micro_symmetry_model: The small model used in weight symmetry tests.
    """
    # Deliberately set the weights of neurons 3 and 4 in dense_1.
    weights = micro_symmetry_model.get_weights()
    # Weights in.
    weights[0][:, 2:4] = np.ones((2, 2))
    # Biases in.
    weights[1][2:4] = np.ones((2,))
    # Weights out.
    weights[2][2:4, :] = np.ones((2, 3))
    micro_symmetry_model.set_weights(weights)
    assert are_symmetric_dense_neurons(
        micro_symmetry_model,
        'dense_1',
        {2, 3}
    )
    # Cause asymmetry in weights out.
    weights[2][2:4, :] = np.array([[1, 1, 1], [2, 2, 2]])
    micro_symmetry_model.set_weights(weights)
    assert not are_symmetric_dense_neurons(
        micro_symmetry_model,
        'dense_1',
        {2, 3}
    )


def test_are_symmetric_dense_neurons_constant_initialized_trained(
        micro_symmetry_model: Sequential,
        micro_symmetry_dataset: tuple[np.ndarray, np.ndarray]) -> None:
    """Tests that a model whose weights were initialized to a constant has
    weight symmetry even after training.

    :param micro_symmetry_model: The small model used in weight symmetry tests.
    :param micro_symmetry_dataset: The training dataset for the model.
    """
    weights = micro_symmetry_model.get_weights()
    for idx, layer_weights in enumerate(weights):
        weights[idx] = np.ones_like(layer_weights)
    micro_symmetry_model.set_weights(weights)
    assert are_symmetric_dense_neurons(
        micro_symmetry_model, 'dense_1', set(range(MICRO_HIDDEN_LEN)))
    assert are_symmetric_dense_neurons(
        micro_symmetry_model, 'dense_2', set(range(MICRO_OUTPUT_LEN)))
    X_train, y_train = micro_symmetry_dataset
    micro_symmetry_model.fit(X_train, y_train, epochs=5)
    assert are_symmetric_dense_neurons(
        micro_symmetry_model, 'dense_1', set(range(MICRO_HIDDEN_LEN)))


def test_are_symmetric_dense_neurons_multi_output_is_asymmetric(
        micro_symmetry_model: Sequential,
        micro_symmetry_dataset: tuple[np.ndarray, np.ndarray]) -> None:
    """Tests that a multi-output model does not, in general, have symmetric
    weights in the last layer, even if they were symmetric before training.
    Note that, since the hidden layer nodes are symmetric (see
    test_are_symmetric_dense_neurons_constant_initialized_trained()), the rows
    in the weight matrix of the final layer will be identical. This is not
    shown in this test because it was already covered in the aforementioned.
    Also note that the single output case is already demonstrated to be
    symmetric (by definition, since there is only one neuron) in
    test_are_symmetric_dense_neurons_one_neuron().

    :param micro_symmetry_model: The small model used in weight symmetry tests.
    :param micro_symmetry_dataset: The training dataset for the model.
    """
    weights = micro_symmetry_model.get_weights()
    for idx, layer_weights in enumerate(weights):
        weights[idx] = np.ones_like(layer_weights)
    micro_symmetry_model.set_weights(weights)
    assert are_symmetric_dense_neurons(
        micro_symmetry_model, 'dense_2', set(range(MICRO_OUTPUT_LEN)))
    X_train, y_train = micro_symmetry_dataset
    micro_symmetry_model.fit(X_train, y_train, epochs=5)
    assert not are_symmetric_dense_neurons(
        micro_symmetry_model, 'dense_2', set(range(MICRO_OUTPUT_LEN)))


def test_are_symmetric_dense_neurons_random_initialization(
        micro_symmetry_model: Sequential) -> None:
    """Tests that a model whose weights were randomly initialized causes no
    weights symmetry.

    :param micro_symmetry_model: The small model used in weight symmetry tests.
    """
    # Pick two arbitrary neuron indices.
    neuron_indices = {0, 1}
    assert not are_symmetric_dense_neurons(
        micro_symmetry_model, 'dense_1', neuron_indices)
    assert not are_symmetric_dense_neurons(
        micro_symmetry_model, 'dense_2', neuron_indices)


def test_are_symmetric_dense_neurons_raises_error(
        mnist_model: Sequential) -> None:
    """Tests that are_symmetric_dense_neurons raises a NotADenseLayerError when
    supplied with a non-dense layer.

    :param mnist_model: The baseline model.
    """
    with pytest.raises(NotADenseLayerError):
        _ = are_symmetric_dense_neurons(mnist_model, 'flatten', {0})
    mnist_model.add(Flatten(name='flatten_out'))
    with pytest.raises(NotADenseLayerError):
        _ = are_symmetric_dense_neurons(mnist_model, 'dense_2', {0})


def test_expand_dense_layer_increases_layer_size(
        mnist_model: Sequential) -> None:
    """Tests that expand_dense_layer increases the size of the given layer.

    :param mnist_model: The baseline model.
    """
    num_new_neurons = 5
    expanded = expand_dense_layer(mnist_model, 'dense_1', num_new_neurons)
    assert expanded.get_layer('dense_1').units == \
           mnist_model.get_layer('dense_1').units + num_new_neurons


def test_expand_dense_layer_on_output_layer(
        mnist_model: Sequential) -> None:
    """Tests that expand_dense_layer can increase the size of the output layer.

    :param mnist_model: The baseline model.
    """
    num_new_neurons = 5
    expanded = expand_dense_layer(mnist_model, 'dense_2', num_new_neurons)
    assert expanded.get_layer('dense_2').units == \
           mnist_model.get_layer('dense_2').units + num_new_neurons


def test_expand_dense_layer_all_zero_same_output(
        mnist_model: Sequential) -> None:
    """Tests that the all zero strategy does not change model output.

    :param mnist_model: The baseline model.
    """
    num_new_neurons = 5
    expanded = expand_dense_layer(
        mnist_model, 'dense_1', num_new_neurons, strategy=STRATEGY_ALL_ZERO)
    batch = np.ones((2, *MNIST_IMAGE_SHAPE))
    assert np.isclose(mnist_model(batch), expanded(batch)).all()


def test_expand_dense_layer_all_zero_causes_weight_symmetry(
        mnist_model: Sequential,
        mnist_dataset: tuple[tuple[np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray]]) -> None:
    """Tests that the all zero strategy causes weight symmetry when the model
    is fine-tuned.

    :param mnist_model: The baseline model.
    :param mnist_dataset: The MNIST dataset.
    """
    (X_train, y_train), _ = mnist_dataset
    num_old_neurons = mnist_model.get_layer('dense_1').units
    num_new_neurons = 5
    expanded = expand_dense_layer(
        mnist_model, 'dense_1', num_new_neurons, strategy=STRATEGY_ALL_ZERO)
    expanded.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['sparse_categorical_accuracy'])
    expanded.fit(X_train, y_train, epochs=5)
    assert are_symmetric_dense_neurons(
        expanded,
        'dense_1',
        set(range(num_old_neurons, num_old_neurons + num_new_neurons))
    )


def test_expand_dense_layer_output_zero_same_output(
        mnist_model: Sequential) -> None:
    """Tests that the output zero strategy does not change model output.

    :param mnist_model: The baseline model.
    """
    num_new_neurons = 5
    expanded = expand_dense_layer(
        mnist_model,
        'dense_1',
        num_new_neurons,
        strategy=STRATEGY_OUTPUT_ZERO)
    batch = np.ones((2, *MNIST_IMAGE_SHAPE))
    assert np.isclose(mnist_model(batch), expanded(batch)).all()


def test_expand_dense_layer_output_zero_same_output_trained_model(
        mnist_model: Sequential,
        mnist_dataset: tuple[tuple[np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray]]) -> None:
    """Tests that the output zero strategy does not change model output, even
    when the model has been pretrained.

    :param mnist_model: The baseline model.
    :param mnist_dataset: The MNIST dataset.
    """
    (X_train, y_train), (X_test, y_test) = mnist_dataset
    mnist_model.fit(X_train, y_train, epochs=5)
    baseline_eval = mnist_model.evaluate(X_test, y_test)
    num_new_neurons = 5
    expanded = expand_dense_layer(
        mnist_model,
        'dense_1',
        num_new_neurons,
        strategy=STRATEGY_OUTPUT_ZERO)
    expanded.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['sparse_categorical_accuracy'])
    expanded_eval = expanded.evaluate(X_test, y_test)
    assert np.isclose(baseline_eval, expanded_eval).all()


def test_expand_dense_layer_output_zero_no_weight_symmetry(
        mnist_model: Sequential,
        mnist_dataset: tuple[tuple[np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray]]) -> None:
    """Tests that the output zero strategy does not cause weight symmetry when
    the model is fine-tuned.

    :param mnist_model: The baseline model.
    :param mnist_dataset: The MNIST dataset.
    """
    (X_train, y_train), _ = mnist_dataset
    num_new_neurons = 5
    # Pick two arbitrary new neurons to test for symmetry.
    new_neuron_idx_1 = 1
    new_neuron_idx_2 = 2
    expanded = expand_dense_layer(
        mnist_model,
        'dense_1',
        num_new_neurons,
        strategy=STRATEGY_OUTPUT_ZERO)
    expanded.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['sparse_categorical_accuracy'])
    expanded.fit(X_train, y_train, epochs=5)
    assert not are_symmetric_dense_neurons(
        expanded,
        'dense_1',
        {new_neuron_idx_1, new_neuron_idx_2}
    )


def test_expand_dense_layer_all_random_different_output(
        mnist_model: Sequential) -> None:
    """Tests that the all random strategy changes model output.

    :param mnist_model: The baseline model.
    """
    num_new_neurons = 5
    expanded = expand_dense_layer(
        mnist_model,
        'dense_1',
        num_new_neurons,
        strategy=STRATEGY_ALL_RANDOM)
    batch = np.ones((2, *MNIST_IMAGE_SHAPE))
    assert not np.isclose(mnist_model(batch), expanded(batch)).all()


def test_expand_dense_layer_not_dense_raises_error(
        mnist_model: Sequential) -> None:
    """Tests that expand_dense_layer raises an error when the layer is not
    Dense.

    :param mnist_model: The baseline model.
    """
    with pytest.raises(NotADenseLayerError):
        _ = expand_dense_layer(mnist_model, 'flatten', 1)
    mnist_model.add(Flatten(name='flatten_out'))
    with pytest.raises(NotADenseLayerError):
        _ = expand_dense_layer(mnist_model, 'dense_2', 1)


def test_expand_dense_layer_invalid_strategy_raises_error(
        mnist_model: Sequential) -> None:
    """Tests that expand_dense_layer raises an error when the strategy is
    invalid.

    :param mnist_model: The baseline model.
    """
    with pytest.raises(InvalidExpansionStrategyError):
        _ = expand_dense_layer(mnist_model, 'dense_1', 1, strategy='dne')


def test_expand_dense_layers_increases_layer_sizes(
        multilayer_mnist_model: Sequential) -> None:
    """Tests that expand_dense_layers increases the sizes of the given
    layers.

    :param multilayer_mnist_model: The baseline model.
    """
    dense_layer_names_and_neurons = {'dense_1': 3, 'dense_2': 2}
    expanded = expand_dense_layers(
        multilayer_mnist_model, dense_layer_names_and_neurons)
    for name, num_new_neurons in dense_layer_names_and_neurons.items():
        assert expanded.get_layer(name).units == \
           multilayer_mnist_model.get_layer(name).units + num_new_neurons


def test_expand_dense_layers_maximizes_number_of_random_weights(
        multilayer_mnist_model: Sequential) -> None:
    """Tests that expand_dense_layers randomly initializes the maximum number
    of weights when strategy is output zero.

    Because the new weights are initialized from the input layer to the output
    layer, there should be non-zero weights from new neurons to new neurons.
    Observe that these would be zero if set in the opposite direction, because
    the output zero strategy forces connections from new neurons to existing
    neurons to be zero.

    :param multilayer_mnist_model: The baseline model.
    """
    dense_layer_names_and_neurons = {'dense_1': 3, 'dense_2': 2}
    expanded = expand_dense_layers(
        multilayer_mnist_model,
        dense_layer_names_and_neurons,
        strategy=STRATEGY_OUTPUT_ZERO)
    original_in_layer = multilayer_mnist_model.get_layer('dense_1')
    original_out_layer = multilayer_mnist_model.get_layer('dense_2')
    expanded_out_layer = expanded.get_layer('dense_2')
    weights_and_biases = expanded_out_layer.get_weights()
    weights = weights_and_biases[0]
    weights_from_new_neurons = weights[original_in_layer.units:, :]
    weights_from_new_neurons_to_new_neurons = \
        weights_from_new_neurons[:, original_out_layer.units:]
    assert weights_from_new_neurons_to_new_neurons.shape == (
        dense_layer_names_and_neurons['dense_1'],
        dense_layer_names_and_neurons['dense_2']
    )
    assert (weights_from_new_neurons_to_new_neurons != 0).all()


def test_expand_dense_layers_all_zero_causes_weight_symmetry(
        multilayer_mnist_model: Sequential,
        mnist_dataset: tuple[tuple[np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray]]) -> None:
    """Tests that expand_dense_layers causes weight symmetry when using the all
    zero strategy.

    :param multilayer_mnist_model: The baseline model.
    :param mnist_dataset: The MNIST dataset.
    """
    (X_train, y_train), _ = mnist_dataset
    dense_layer_names_and_neurons = {'dense_1': 3, 'dense_2': 2}
    expanded = expand_dense_layers(
        multilayer_mnist_model,
        dense_layer_names_and_neurons,
        strategy=STRATEGY_ALL_ZERO)
    expanded.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['sparse_categorical_accuracy'])
    expanded.fit(X_train, y_train, epochs=5)
    num_old_neurons = multilayer_mnist_model.get_layer('dense_1').units
    num_new_neurons = dense_layer_names_and_neurons['dense_1']
    assert are_symmetric_dense_neurons(
        expanded,
        'dense_1',
        set(range(num_old_neurons, num_old_neurons + num_new_neurons))
    )
    num_old_neurons = multilayer_mnist_model.get_layer('dense_2').units
    num_new_neurons = dense_layer_names_and_neurons['dense_2']
    assert are_symmetric_dense_neurons(
        expanded,
        'dense_2',
        set(range(num_old_neurons, num_old_neurons + num_new_neurons))
    )


def test_expand_dense_layers_output_zero_no_weight_symmetry(
        multilayer_mnist_model: Sequential,
        mnist_dataset: tuple[tuple[np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray]]) -> None:
    """Tests that expand_dense_layers does not cause weight symmetry when using
    the output zero strategy.

    :param multilayer_mnist_model: The baseline model.
    :param mnist_dataset: The MNIST dataset.
    """
    (X_train, y_train), _ = mnist_dataset
    dense_layer_names_and_neurons = {'dense_1': 3, 'dense_2': 2}
    expanded = expand_dense_layers(
        multilayer_mnist_model,
        dense_layer_names_and_neurons,
        strategy=STRATEGY_OUTPUT_ZERO)
    expanded.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['sparse_categorical_accuracy'])
    expanded.fit(X_train, y_train, epochs=5)
    # Pick two arbitrary new neurons to test for symmetry.
    new_neuron_idx_1 = 2
    new_neuron_idx_2 = 3
    assert not are_symmetric_dense_neurons(
        expanded,
        'dense_1',
        {new_neuron_idx_1, new_neuron_idx_2}
    )
    assert not are_symmetric_dense_neurons(
        expanded,
        'dense_2',
        {new_neuron_idx_1, new_neuron_idx_2}
    )


def test_clone_layer_uses_previous_config() -> None:
    """Tests that clone_layer uses previous layer configuration."""
    units = 10
    activation = 'relu'
    use_bias = False
    layer = Dense(units, activation=activation, use_bias=use_bias)
    cloned = clone_layer(layer)
    assert cloned.units == units
    assert cloned.activation.__name__ == activation
    assert cloned.use_bias == use_bias


def test_clone_layer_does_not_copy_weights() -> None:
    """Tests that clone_layer does not copy weights."""
    units = 5
    input_shape = (2,)
    layer = Dense(units, input_shape=input_shape)
    _ = Sequential([layer])
    cloned = clone_layer(layer)
    _ = Sequential([cloned])
    assert not np.array_equal(layer.get_weights(), cloned.get_weights())


def test_clone_layer_replaces_config() -> None:
    """Tests that clone_layer replaces configuration parameters when new values
    are supplied."""
    activation = 'relu'
    use_bias = False
    layer = Dense(5, activation=activation, use_bias=use_bias)
    replace = {'units': 10}
    cloned = clone_layer(layer, replace=replace)
    assert cloned.units == replace['units']
    assert cloned.activation.__name__ == activation
    assert cloned.use_bias == use_bias
