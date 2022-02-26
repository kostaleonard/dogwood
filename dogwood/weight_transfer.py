"""Contains functions to transfer knowledge from pretrained model weights."""

from typing import Dict
from tensorflow.keras.models import Sequential

STRATEGY_ALL_ZERO = 'all_zero'
STRATEGY_OUTPUT_ZERO = 'output_zero'
STRATEGY_ALL_RANDOM = 'all_random'


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
    # TODO add support for arbitrary architectures


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
    # TODO add support for arbitrary architectures
