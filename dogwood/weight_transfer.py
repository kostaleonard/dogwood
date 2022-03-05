"""Contains functions to transfer knowledge from pretrained model weights."""
# pylint: disable=no-name-in-module

from __future__ import annotations
from typing import Any
from itertools import combinations
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.initializers import GlorotUniform
from dogwood.errors import NotADenseLayerError, InvalidExpansionStrategyError

STRATEGY_ALL_ZERO = 'all_zero'
STRATEGY_OUTPUT_ZERO = 'output_zero'
STRATEGY_ALL_RANDOM = 'all_random'


def are_symmetric_dense_neurons(
        model: Sequential,
        layer_name: str,
        neuron_indices: set[int]) -> bool:
    """Returns True if all of the given neurons are symmetric with each other.

    We define "neuron symmetry" as two or more neurons computing the (exact)
    same function on every input. In other words, on any arbitrary signal, the
    neurons receive the same input and propagate the same output--all weights
    and biases from the previous layer must be the same, and all weights to the
    next layer must be the same with respect to those specific neurons. Neuron
    symmetry occurs when the weights of a neural network are initialized to a
    constant value, not randomly.

    :param model: The model whose weights to examine.
    :param layer_name: The name of the dense layer in which the neurons reside.
    :param neuron_indices: The indices of the neurons in the dense layer. If
        only one index is provided, the result will always be True.
    :return: True if all of the given neurons are symmetric with each other,
        False otherwise.
    """
    # pylint: disable=too-many-locals
    layer_in, layer_out = _get_layer_input_and_output_by_name(
        model, layer_name)
    if not isinstance(layer_in, Dense):
        raise NotADenseLayerError
    if layer_out and not isinstance(layer_out, Dense):
        raise NotADenseLayerError
    weights_and_biases_in = layer_in.get_weights()
    weights_in = weights_and_biases_in[0]
    biases_in = weights_and_biases_in[1]
    # Outgoing biases do not contribute to symmetry.
    weights_out = np.array([]) if not layer_out else layer_out.get_weights()[0]
    weights_in_neurons = [weights_in[:, idx] for idx in neuron_indices]
    biases_in_neurons = [biases_in[idx] for idx in neuron_indices]
    weights_out_neurons = [] if not layer_out else [
        weights_out[idx, :] for idx in neuron_indices]
    for weights_in_neurons_1, weights_in_neurons_2 in combinations(
            weights_in_neurons, 2):
        if not np.isclose(weights_in_neurons_1, weights_in_neurons_2).all():
            return False
    for biases_in_neurons_1, biases_in_neurons_2 in combinations(
            biases_in_neurons, 2):
        if not np.isclose(biases_in_neurons_1, biases_in_neurons_2):
            return False
    for weights_out_neurons_1, weights_in_neurons_2 in combinations(
            weights_out_neurons, 2):
        if not np.isclose(weights_out_neurons_1, weights_in_neurons_2).all():
            return False
    return True


def expand_dense_layer(
        model: Sequential,
        layer_name: str,
        num_new_neurons: int,
        strategy: str = STRATEGY_OUTPUT_ZERO) -> Sequential:
    """Returns a new model with additional neurons in the given dense layer.

    Used to update only a single layer. While expanding several layers in a
    network with repeated calls to this function is possible and will not cause
    weight symmetry problems, more of the network can be randomly initialized
    when using expand_dense_layers(), which does this update intelligently.

    Here are a few notes regarding random initialization:
        1. GlorotUniform initialization depends on the size of the input, so we
            initialize with the full size of the new layer, then subset to the
            new weights.
        2. New bias units are set to 0, the default value for Dense layers.

    :param model: The model whose base architecture and weights to use. No
        other elements of the architecture are changed; all weights are copied
        to the new model.
    :param layer_name: The name of the dense layer to expand.
    :param num_new_neurons: The number of neurons to add to the layer.
    :param strategy: The strategy with which to populate the new weights into
        and out of the additional neurons. Please see #15 for gradient
        calculations to support the discussion below. Choices are as follows.
            STRATEGY_ALL_ZERO: Fill all new weights and biases with 0. This is
                not recommended, since it will cause weight symmetry and impair
                model performance gain during fine-tuning. Preserves the
                performance of the neural network exactly as it was before
                adding neurons.
            STRATEGY_OUTPUT_ZERO: Fill all new weights and biases into new
                neurons randomly using Glorot Uniform; fill all new weights
                out of new neurons with zeros. This option prevents weight
                symmetry, but preserves the performance of the neural network
                exactly as it was before adding the neurons.
            STRATEGY_ALL_RANDOM: Fill all new weights and biases randomly using
                Glorot Uniform. Performance of the neural network will be
                altered.
    :return: A new model with additional neurons in the given dense layer. The
        output model will use the weights of the input model, but performance
        may not be identical based on choice of strategy.
    """
    # pylint: disable=too-many-locals
    if strategy not in {
            STRATEGY_ALL_ZERO, STRATEGY_OUTPUT_ZERO, STRATEGY_ALL_RANDOM}:
        raise InvalidExpansionStrategyError
    layer_in, layer_out = _get_layer_input_and_output_by_name(
        model, layer_name)
    if not isinstance(layer_in, Dense):
        raise NotADenseLayerError
    if layer_out and not isinstance(layer_out, Dense):
        raise NotADenseLayerError
    # Create new model, add new weights.
    # Layer weights can only be set after the layer has been added to a model.
    expanded = Sequential()
    glorot = GlorotUniform()
    for layer in model.layers:
        if layer == layer_in:
            weights_and_biases = layer.get_weights()
            weights = weights_and_biases[0]
            biases = weights_and_biases[1]
            if strategy == STRATEGY_ALL_ZERO:
                new_weights = np.concatenate(
                    (weights, np.zeros((len(weights), num_new_neurons))),
                    axis=1)
            else:
                new_weights_shape = (
                    len(weights), weights.shape[1] + num_new_neurons)
                initialization = glorot(new_weights_shape)[:, :num_new_neurons]
                new_weights = np.concatenate((weights, initialization), axis=1)
            new_biases = np.concatenate(
                (biases, np.zeros(num_new_neurons)), axis=0)
            replace = {'units': layer.units + num_new_neurons}
            new_layer = clone_layer(layer, replace=replace)
            expanded.add(new_layer)
            new_layer.set_weights([new_weights, new_biases])
        elif layer == layer_out:
            weights_and_biases = layer.get_weights()
            weights = weights_and_biases[0]
            biases = weights_and_biases[1]
            if strategy == STRATEGY_ALL_RANDOM:
                new_weights_shape = (
                    len(weights) + num_new_neurons, weights.shape[1])
                initialization = glorot(new_weights_shape)[:num_new_neurons, :]
                new_weights = np.concatenate((weights, initialization), axis=0)
            else:
                new_weights = np.concatenate(
                    (weights, np.zeros((num_new_neurons, weights.shape[1]))),
                    axis=0)
            new_layer = clone_layer(layer)
            expanded.add(new_layer)
            new_layer.set_weights([new_weights, biases])
        else:
            new_layer = clone_layer(layer)
            expanded.add(new_layer)
            new_layer.set_weights(layer.get_weights())
    return expanded


def expand_dense_layers(
        model: Sequential,
        layer_names_and_neurons: dict[str, int],
        strategy: str = STRATEGY_OUTPUT_ZERO) -> Sequential:
    """Returns a new model with additional neurons in the given dense layers.

    Used to update multiple layers in a network. The layers are traversed by
    index, so that new neurons are added to earlier layers before later layers.
    This increases the number of weights that can be randomly initialized.

    :param model: The model whose base architecture and weights to use. No
        other elements of the architecture are changed; all weights are copied
        to the new model.
    :param layer_names_and_neurons: A dictionary whose keys are the names
        of the dense layers to expand and whose values are the number of
        neurons to add to each layer.
    :param strategy: The strategy with which to populate the new weights into
        and out of the additional neurons. Options are identical to those in
        expand_dense_layer().
    :return: A new model with additional neurons in the given dense layers. The
        output model will use the weights of the input model, but performance
        may not be identical based on choice of strategy.
    """
    expanded = model
    for layer in model.layers:
        if layer.name in layer_names_and_neurons:
            expanded = expand_dense_layer(
                expanded,
                layer.name,
                layer_names_and_neurons[layer.name],
                strategy=strategy)
    return expanded


def _get_layer_input_and_output_by_name(
        model: Sequential, layer_name: str) -> tuple[Layer, Layer | None]:
    """Returns the requested layer and the subsequent layer, if it exists.

    :param model: The model in which to find the layer.
    :param layer_name: The name of the layer to find.
    :return: The requested layer and the subsequent layer, if it exists.
    """
    layer_names = [layer.name for layer in model.layers]
    layer_idx = layer_names.index(layer_name)
    layer_in = model.layers[layer_idx]
    layer_out = None if layer_idx == len(model.layers) - 1 else \
        model.layers[layer_idx + 1]
    return layer_in, layer_out


def clone_layer(
        layer: Layer,
        replace: dict[str, Any] | None = None) -> Layer:
    """Returns a new instance of the layer's class with the same configuration.

    :param layer: The layer to clone.
    :param replace: A dictionary of configurations to replace. The keys are
        generally arguments to the layer class's __init__(), and the values
        correspond to the configuration keys.
    :return: A new instance of the layer's class with the same configuration.
        The cloned layer does not have the same weights as the original.
    """
    # This implementation is based on tensorflow.keras.models in TFv2.8.
    if replace:
        config = {**layer.get_config(), **replace}
    else:
        config = layer.get_config()
    return layer.__class__.from_config(config)
