"""Contains the PretrainingPool class."""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from tensorflow.keras.models import Model
from mlops.model.versioned_model import VersionedModel
from mlops.dataset.versioned_dataset import VersionedDataset

DEFAULT_DIRNAME = os.path.join(Path.home(), '.dogwood', 'pretrained')
OPEN_SOURCE_MODELS = {'VGG16'}
DEFAULT_MODELS = 'default'


class PretrainingPool:
    """Represents a pool of pretrained models from which to transfer weights.

    When the pool is instantiated, it searches the filesystem for pretrained
    models. If it cannot find any, it downloads the default open source models.
    This behavior is similar to how keras stores models and datasets. Custom
    models can (and should) be added to the pool as storage costs allow to
    improve the performance that can be achieved through transfer learning.

    Pool structure:
    dirname/
        model_name_1/
            model_name_1.h5
            X_train.npy
            y_train.npy
        model_name_2/
            model_name_2.h5
            X_train.npy
            y_train.npy
        ...
    """

    def __init__(
            self,
            dirname: str = DEFAULT_DIRNAME,
            with_models: str | set[str] | None = DEFAULT_MODELS) -> None:
        """Instantiates the object.

        The given path is, if necessary, created and filled with the specified
        models.

        :param dirname: The path to the pool. It is created if it does not
            exist.
        :param with_models: The models with which to instantiate the pool. Can
            be any of the following:
                str: A string indicating any one of the open source models, or
                    the string 'default' indicating the default open source
                    models.
                set[str]: A set of strings indicating open source models.
                None: No models.
        """
        # TODO

    def add_model(self,
                  model: Model,
                  X_train: np.ndarray,
                  y_train: np.ndarray) -> None:
        """Adds the model to the pool.

        :param model: The model to add to the pool.
        :param X_train: The model's training features.
        :param y_train: The model's training labels.
        """
        # TODO

    def add_versioned_model(self,
                            model: VersionedModel,
                            dataset: VersionedDataset) -> None:
        """Adds the versioned model to the pool.

        :param model: The versioned model.
        :param dataset: The versioned dataset.
        """
        # TODO test this
        # TODO we potentially want a special representation.
        # TODO if no X_train/y_train, make user specify attr names.
        # self.add_model(model.model, dataset.X_train, dataset.y_train)

    def get_pretrained_model(self,
                             model: Model,
                             X_train: np.ndarray,
                             y_train: np.ndarray) -> Model:
        """Returns a new instance of the given model with pretrained weights.

        Creates a new model of the same architecture as the input, but with the
        best possible pretrained weights based on the models in the pool.

        :param model: The model to pretrain.
        :param X_train: The model's training features.
        :param y_train: The model's trainined labels.
        :return: A new instance of the given model with pretrained weights.
        """
        # TODO copy the first N layers and expand as necessary, maximizing performance on the training dataset? What about different architectures?
