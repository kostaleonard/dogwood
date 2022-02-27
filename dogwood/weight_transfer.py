"""Contains functions to transfer knowledge from pretrained model weights."""
# TODO add support for arbitrary architectures, not just Sequential

from typing import Dict, Set
from itertools import combinations
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from dogwood.errors import NotADenseLayerError

STRATEGY_ALL_ZERO = 'all_zero'
STRATEGY_OUTPUT_ZERO = 'output_zero'
STRATEGY_ALL_RANDOM = 'all_random'


def are_symmetric_dense_neurons(
        model: Sequential,
        layer_name: str,
        neuron_indices: Set[int]) -> bool:
    """Returns True if all of the given neurons are symmetric with each other.

    We define "neuron symmetry" as two or more neurons computing the (exact)
    same function on every input. In other words, on any arbitrary signal, the
    neurons receive the same input and propagate the same output--all weights
    from the previous layer must be the same, and all weights to the next layer
    must be the same with respect to those specific neurons. Neuron symmetry
    occurs when the weights of a neural network are initialized to a constant
    value, not randomly.

    :param model: The model whose weights to examine.
    :param layer_name: The name of the dense layer in which the neurons reside.
    :param neuron_indices: The indices of the neurons in the dense layer. If
        only one index is provided, the result will always be True.
    :return: True if all of the given neurons are symmetric with each other,
        False otherwise.
    """
    layer_names = [layer.name for layer in model.layers]
    layer_idx = layer_names.index(layer_name)
    layer_in = model.layers[layer_idx]
    if not isinstance(layer_in, Dense):
        raise NotADenseLayerError
    layer_out = None if layer_idx == len(model.layers) - 1 else \
        model.layers[layer_idx + 1]
    if layer_out and not isinstance(layer_out, Dense):
        raise NotADenseLayerError
    weights_and_biases_in = layer_in.get_weights()
    weights_in = weights_and_biases_in[0]
    biases_in = weights_and_biases_in[1]
    # Outgoing biases do not contribute to symmetry.
    weights_out = [] if not layer_out else layer_out.get_weights()[0]
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
        dense_layer_name: str,
        num_new_neurons: int,
        strategy: str = STRATEGY_OUTPUT_ZERO) -> Sequential:
    """Returns a new model with additional neurons in the given dense layer.

    Used to update only a single layer. While expanding several layers in a
    network with repeated calls to this function is possible and will not cause
    weight symmetry problems, more of the network can be randomly initialized
    when using expand_dense_layers(), which does this update intelligently.

    :param model: The model whose base architecture and weights to use. No
        other elements of the architecture are changed; all weights are copied
        to the new model.
    :param dense_layer_name: The name of the dense layer to expand.
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
    # TODO raise error if not dense layer
    return model


def expand_dense_layers(
        model: Sequential,
        dense_layer_names_and_neurons: Dict[str, int],
        strategy: str = STRATEGY_OUTPUT_ZERO) -> Sequential:
    """Returns a new model with additional neurons in the given dense layers.

    Used to update multiple layers in a network. The layers are traversed by
    index, so that new neurons are added to earlier layers before later layers.
    This increases the number of weights that can be randomly initialized.

    :param model: The model whose base architecture and weights to use. No
        other elements of the architecture are changed; all weights are copied
        to the new model.
    :param dense_layer_names_and_neurons: A dictionary whose keys are the names
        of the dense layers to expand and whose values are the number of
        neurons to add to each layer.
    :param strategy: The strategy with which to populate the new weights into
        and out of the additional neurons. Options are identical to those in
        expand_dense_layer().
    :return: A new model with additional neurons in the given dense layers. The
        output model will use the weights of the input model, but performance
        may not be identical based on choice of strategy.
    """
    # TODO define in terms of expand_dense_layer()
    return model
