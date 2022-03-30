"""Contains the PretrainingPool class."""
# pylint: disable=no-name-in-module

from __future__ import annotations
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from mlops.model.versioned_model_builder import VersionedModelBuilder
from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder, \
    STRATEGY_COPY_ZIP
from mlops.model.versioned_model import VersionedModel
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.dataset.pathless_versioned_dataset_builder import \
    PathlessVersionedDatasetBuilder
from mlops.errors import PublicationPathAlreadyExistsError
from dogwood.errors import PretrainingPoolAlreadyContainsModelError, \
    NoSuchOpenSourceModelError, UnrecognizedTrainingDatasetError
from dogwood.pretraining.imagenet_data_processor import ImageNetDataProcessor
from dogwood.pretraining.mini_imagenet_loader import download_mini_imagenet, \
    MINI_IMAGENET_DIRNAME
from dogwood import DOGWOOD_DIR

PRETRAINED_DIRNAME = os.path.join(DOGWOOD_DIR, 'pretrained')
MODEL_VGG16 = 'VGG16'
VGG16_VERSION = 'v1'
OPEN_SOURCE_MODELS = {MODEL_VGG16}
DEFAULT_MODELS = 'default'
DATASET_MINI_IMAGENET = 'imagenet-mini'
MINI_IMAGENET_VERSION = 'v1'
MODEL_DATASETS = {MODEL_VGG16: DATASET_MINI_IMAGENET}
TAG_USER_MODEL = 'user'


class PretrainingPool:
    """Represents a pool of pretrained models from which to transfer weights.

    When the pool is instantiated, it searches the filesystem for pretrained
    models. If it cannot find any, it downloads the default open source models.
    This behavior is similar to how keras stores models and datasets. Custom
    models can (and should) be added to the pool as storage costs allow to
    improve the performance that can be achieved through transfer learning.

    Pool structure:
    dirname/
        models/
            model_name_1/version/
                model.h5 (the saved model)
                meta.json (metadata)
            ...
        datasets/
            dataset_name_1/version/
                X_train.npy (and other feature tensors by their given names)
                y_train.npy (and other label tensors by their given names)
                data_processor.pkl (DataProcessor object)
                meta.json (metadata)
                raw.tar.bz2 (bz2-zipped directory with the raw dataset files)
            ...
    """

    def __init__(
            self,
            dirname: str = PRETRAINED_DIRNAME,
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
        self.models_dirname = os.path.join(dirname, 'models')
        self.datasets_dirname = os.path.join(dirname, 'datasets')
        if isinstance(with_models, set):
            if with_models - OPEN_SOURCE_MODELS:
                raise NoSuchOpenSourceModelError
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
        Path(self.models_dirname).mkdir(parents=True, exist_ok=True)
        Path(self.datasets_dirname).mkdir(parents=True, exist_ok=True)
        for model_name in self.with_models:
            model_dataset = MODEL_DATASETS[model_name]
            # Populate dataset.
            if model_dataset == DATASET_MINI_IMAGENET:
                self._populate_mini_imagenet()
            # Populate model.
            if model_name == MODEL_VGG16:
                self._populate_vgg16()

    def _populate_mini_imagenet(self) -> None:
        """Instantiates the mini ImageNet dataset, if it does not exist."""
        publication_path = os.path.join(
            self.datasets_dirname, DATASET_MINI_IMAGENET)
        if not os.path.exists(
                os.path.join(publication_path, MINI_IMAGENET_VERSION)):
            processor = ImageNetDataProcessor()
            with TemporaryDirectory() as tempdir:
                download_mini_imagenet(tempdir)
                dataset_path = os.path.join(tempdir, MINI_IMAGENET_DIRNAME)
                builder = VersionedDatasetBuilder(dataset_path, processor)
                builder.publish(publication_path,
                                version=MINI_IMAGENET_VERSION,
                                dataset_copy_strategy=STRATEGY_COPY_ZIP,
                                tags=['image'])

    def _populate_vgg16(self) -> None:
        """Instantiates the VGG16 model, if it does not exist."""
        publication_path = os.path.join(self.models_dirname, MODEL_VGG16)
        if not os.path.exists(os.path.join(publication_path, VGG16_VERSION)):
            dataset_path = os.path.join(
                self.datasets_dirname,
                DATASET_MINI_IMAGENET,
                MINI_IMAGENET_VERSION)
            dataset = VersionedDataset(dataset_path)
            model = VGG16()
            builder = VersionedModelBuilder(dataset, model)
            builder.publish(publication_path,
                            version=VGG16_VERSION,
                            tags=['image'])

    def add_model(self,
                  model: Model,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  dataset_name: str = 'dataset') -> None:
        """Adds the model to the pool.

        :param model: The model to add to the pool.
        :param X_train: The model's training features.
        :param y_train: The model's training labels.
        :param dataset_name: The name of the dataset. If a dataset with this
            name already exists, raises a PublicationPathAlreadyExistsError.
        """
        features = {'X_train': X_train}
        labels = {'y_train': y_train}
        dataset_builder = PathlessVersionedDatasetBuilder(features, labels)
        dataset_publication_path = os.path.join(
            self.datasets_dirname, dataset_name)
        try:
            dataset_builder.publish(dataset_publication_path,
                                    name=dataset_name,
                                    version='v1',
                                    tags=[TAG_USER_MODEL])
        except PublicationPathAlreadyExistsError:
            # Allow dataset reuse between models.
            pass
        dataset = VersionedDataset(os.path.join(
            dataset_publication_path, 'v1'))
        model_builder = VersionedModelBuilder(dataset, model)
        model_publication_path = os.path.join(self.models_dirname, model.name)
        try:
            model_builder.publish(model_publication_path,
                                  name=model.name,
                                  version='v1',
                                  tags=[TAG_USER_MODEL])
        except PublicationPathAlreadyExistsError as err:
            raise PretrainingPoolAlreadyContainsModelError from err

    def add_versioned_model(self,
                            model: VersionedModel,
                            dataset: VersionedDataset) -> None:
        """Adds the versioned model to the pool.

        :param model: The versioned model.
        :param dataset: The versioned dataset.
        """
        if not hasattr(dataset, 'X_train') or not hasattr(dataset, 'y_train'):
            raise UnrecognizedTrainingDatasetError
        dataset_publication_path = os.path.join(
            self.datasets_dirname, dataset.name)
        try:
            dataset.republish(dataset_publication_path)
        except PublicationPathAlreadyExistsError:
            # Allow dataset reuse between models.
            pass
        model_publication_path = os.path.join(self.models_dirname, model.name)
        try:
            model.republish(model_publication_path)
        except PublicationPathAlreadyExistsError as err:
            raise PretrainingPoolAlreadyContainsModelError from err

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
        # Determine model architecture similarity.
        # Transfer knowledge from similar tasks and architectures.
        raise NotImplementedError
