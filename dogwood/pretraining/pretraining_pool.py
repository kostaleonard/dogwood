"""Contains the PretrainingPool class."""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from mlops.model.versioned_model import VersionedModel
from mlops.dataset.versioned_dataset import VersionedDataset
from dogwood.errors import PretrainingPoolAlreadyContainsModelError, \
    NoSuchOpenSourceModelError

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
        self.dirname = dirname
        Path(dirname).mkdir(parents=True, exist_ok=True)
        if isinstance(with_models, set):
            self.with_models = with_models
        elif isinstance(with_models, str):
            if with_models == 'default':
                self.with_models = OPEN_SOURCE_MODELS
            elif with_models in OPEN_SOURCE_MODELS:
                self.with_models = {with_models}
            else:
                raise NoSuchOpenSourceModelError
        else:
            self.with_models = set()
        self._populate_open_source_models()

    def _populate_open_source_models(self) -> None:
        """Instantiates all of the open source models in the pool.

        Creates all of the files and directories for each open source model in
        the pool. If the files already exist for a model, then no action is
        taken; if the files do not exist, they are downloaded.
        """
        for model_name in self.with_models:
            model_dirname = os.path.join(self.dirname, model_name)
            if not os.path.exists(model_dirname):
                os.mkdir(model_dirname)
            model_path = os.path.join(model_dirname, f'{model_name}.h5')
            X_train_path = os.path.join(model_dirname, 'X_train.npy')
            y_train_path = os.path.join(model_dirname, 'y_train.npy')
            if not os.path.exists(model_path):
                pass
            # TODO download dataset to tempdir
            # TODO publish versioned dataset from files in tempdir

    def add_model(self,
                  model: Model,
                  X_train: np.ndarray,
                  y_train: np.ndarray) -> None:
        """Adds the model to the pool.

        :param model: The model to add to the pool.
        :param X_train: The model's training features.
        :param y_train: The model's training labels.
        """
        model_dir = os.path.join(self.dirname, model.name)
        try:
            os.mkdir(model_dir)
        except FileExistsError as exc:
            raise PretrainingPoolAlreadyContainsModelError from exc
        model_path = os.path.join(model_dir, f'{model.name}.h5')
        X_train_path = os.path.join(model_dir, 'X_train.npy')
        y_train_path = os.path.join(model_dir, 'y_train.npy')
        model.save(model_path)
        np.save(X_train_path, X_train)
        np.save(y_train_path, y_train)

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
        # Determine task similarity.
        # TODO compare task embeddings or marginal/conditional distributions?
        # TODO how can we compute task embeddings on arbitrary networks/datasets?
        # Determine model architecture similarity.
        # TODO
        # Transfer knowledge from similar tasks and architectures.
        # TODO copy the first N layers and expand as necessary, maximizing performance on the training dataset? What about different architectures?
        return model
