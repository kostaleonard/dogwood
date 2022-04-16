"""Contains the PretrainingPool class."""
# pylint: disable=no-name-in-module

from __future__ import annotations
from typing import Any
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from packaging.version import parse as parse_version
import numpy as np
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input as preprocess_input_vgg16,
)
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB7,
    preprocess_input as preprocess_input_effnet,
)
from mlops.artifact.versioned_artifact import VersionedArtifact
from mlops.model.versioned_model_builder import VersionedModelBuilder
from mlops.model.versioned_model import VersionedModel
from mlops.dataset.versioned_dataset_builder import (
    VersionedDatasetBuilder,
    STRATEGY_COPY_ZIP,
)
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.dataset.pathless_versioned_dataset_builder import (
    PathlessVersionedDatasetBuilder,
)
from mlops.errors import PublicationPathAlreadyExistsError
from dogwood.errors import (
    PretrainingPoolAlreadyContainsModelError,
    NoSuchOpenSourceModelError,
    UnrecognizedTrainingDatasetError,
    PretrainingPoolCannotCompileCustomModelError,
    ArtifactNotInPoolError,
)
from dogwood.pretraining.imagenet_data_processor import ImageNetDataProcessor
from dogwood.pretraining.mini_imagenet_loader import (
    download_mini_imagenet,
    MINI_IMAGENET_DIRNAME,
)
from dogwood import DOGWOOD_DIR

PRETRAINED_DIRNAME = os.path.join(DOGWOOD_DIR, "pretrained")
MODEL_VGG16 = "VGG16"
VGG16_VERSION = "v1"
VGG_INPUT_SHAPE = (224, 224)
MODEL_EFFICIENTNETB7 = "EfficientNetB7"
EFFICIENTNETB7_VERSION = "v1"
EFFICIENTNETB7_INPUT_SHAPE = (600, 600)
OPEN_SOURCE_MODELS = {MODEL_VGG16, MODEL_EFFICIENTNETB7}
DEFAULT_MODELS = "default"
DATASET_MINI_IMAGENET = "imagenet-mini"
MINI_IMAGENET_VERSION = "v1"
MODEL_DATASETS = {
    MODEL_VGG16: DATASET_MINI_IMAGENET,
    MODEL_EFFICIENTNETB7: DATASET_MINI_IMAGENET,
}
TAG_USER_MODEL = "user"


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
        with_models: str | set[str] | None = DEFAULT_MODELS,
    ) -> None:
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
        self.models_dirname = os.path.join(dirname, "models")
        self.datasets_dirname = os.path.join(dirname, "datasets")
        if isinstance(with_models, set):
            if with_models - OPEN_SOURCE_MODELS:
                raise NoSuchOpenSourceModelError
            self.with_models = with_models
        elif isinstance(with_models, str):
            if with_models == "default":
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
            elif model_name == MODEL_EFFICIENTNETB7:
                self._populate_efficientnetb7()

    def _populate_mini_imagenet(self) -> None:
        """Instantiates the mini ImageNet dataset, if it does not exist."""
        publication_path = os.path.join(
            self.datasets_dirname, DATASET_MINI_IMAGENET
        )
        if not os.path.exists(
            os.path.join(publication_path, MINI_IMAGENET_VERSION)
        ):
            processor = ImageNetDataProcessor()
            with TemporaryDirectory() as tempdir:
                download_mini_imagenet(tempdir)
                dataset_path = os.path.join(tempdir, MINI_IMAGENET_DIRNAME)
                builder = VersionedDatasetBuilder(dataset_path, processor)
                builder.publish(
                    publication_path,
                    name=DATASET_MINI_IMAGENET,
                    version=MINI_IMAGENET_VERSION,
                    dataset_copy_strategy=STRATEGY_COPY_ZIP,
                    tags=["image"],
                )

    def _populate_vgg16(self) -> None:
        """Instantiates the VGG16 model, if it does not exist."""
        publication_path = os.path.join(self.models_dirname, MODEL_VGG16)
        if not os.path.exists(os.path.join(publication_path, VGG16_VERSION)):
            dataset_path = os.path.join(
                self.datasets_dirname,
                DATASET_MINI_IMAGENET,
                MINI_IMAGENET_VERSION,
            )
            dataset = VersionedDataset(dataset_path)
            model = VGG16()
            builder = VersionedModelBuilder(dataset, model)
            builder.publish(
                publication_path,
                name=MODEL_VGG16,
                version=VGG16_VERSION,
                tags=["image"],
            )

    def _populate_efficientnetb7(self) -> None:
        """Instantiates the EfficientNetB7 model, if it does not exist."""
        publication_path = os.path.join(
            self.models_dirname, MODEL_EFFICIENTNETB7
        )
        if not os.path.exists(
            os.path.join(publication_path, EFFICIENTNETB7_VERSION)
        ):
            dataset_path = os.path.join(
                self.datasets_dirname,
                DATASET_MINI_IMAGENET,
                MINI_IMAGENET_VERSION,
            )
            dataset = VersionedDataset(dataset_path)
            model = EfficientNetB7()
            builder = VersionedModelBuilder(dataset, model)
            builder.publish(
                publication_path,
                name=MODEL_EFFICIENTNETB7,
                version=EFFICIENTNETB7_VERSION,
                tags=["image"],
            )

    def __contains__(self, item: Any) -> bool:
        """Returns True if the item is a dataset or model in the pool.

        :param item: The item to test for membership. It can be one of the
            following.
                str: Artifact (model or dataset) name.
                VersionedModel
                VersionedDataset
        :return: True if the item is a dataset or model in the pool; False
            otherwise.
        """
        artifact_name = None
        if isinstance(item, str):
            artifact_name = item
        elif isinstance(item, VersionedArtifact):
            artifact_name = item.name
        in_pool = False
        if artifact_name:
            get_path_fns = (
                lambda: self.get_model_path(artifact_name),
                lambda: self.get_dataset_path(artifact_name),
            )
            for get_path_fn in get_path_fns:
                try:
                    _ = get_path_fn()
                    in_pool = True
                except ArtifactNotInPoolError:
                    pass
        return in_pool

    def add_model(
        self,
        model: Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        dataset_name: str = "dataset",
    ) -> str:
        """Adds the model to the pool.

        :param model: The model to add to the pool.
        :param X_train: The model's training features.
        :param y_train: The model's training labels.
        :param dataset_name: The name of the dataset. If a dataset with this
            name already exists, it is assumed to be shared between models and
            no error is raised.
        :return: The model's publication path.
        """
        dataset_publication_path = self.add_dataset(
            X_train, y_train, dataset_name=dataset_name
        )
        dataset = VersionedDataset(dataset_publication_path)
        model_builder = VersionedModelBuilder(dataset, model)
        publication_base_path = os.path.join(self.models_dirname, model.name)
        try:
            versioned_path = model_builder.publish(
                publication_base_path,
                name=model.name,
                version="v1",
                tags=[TAG_USER_MODEL],
            )
        except PublicationPathAlreadyExistsError as err:
            raise PretrainingPoolAlreadyContainsModelError from err
        return versioned_path

    def add_versioned_model(
        self, model: VersionedModel, dataset: VersionedDataset
    ) -> str:
        """Adds the versioned model to the pool.

        :param model: The versioned model.
        :param dataset: The versioned dataset.
        :return: The model's publication path.
        """
        republished_dataset_path = self.add_versioned_dataset(dataset)
        model_publication_path = os.path.join(self.models_dirname, model.name)
        try:
            republished_model_path = model.republish(model_publication_path)
        except PublicationPathAlreadyExistsError as err:
            raise PretrainingPoolAlreadyContainsModelError from err
        republished_model = VersionedModel(republished_model_path)
        republished_model.update_metadata(
            {"dataset": republished_dataset_path}
        )
        return republished_model_path

    def add_dataset(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        dataset_name: str = "dataset",
    ) -> str:
        """Adds the dataset to the pool.

        :param X_train: The training features.
        :param y_train: The training labels.
        :param dataset_name: The name of the dataset. If a dataset with this
            name already exists, it is assumed to be shared between models and
            no error is raised.
        :return: The dataset's publication path.
        """
        features = {"X_train": X_train}
        labels = {"y_train": y_train}
        dataset_builder = PathlessVersionedDatasetBuilder(features, labels)
        publication_base_path = os.path.join(
            self.datasets_dirname, dataset_name
        )
        versioned_path = os.path.join(publication_base_path, "v1")
        try:
            dataset_builder.publish(
                publication_base_path,
                name=dataset_name,
                version="v1",
                tags=[TAG_USER_MODEL],
            )
        except PublicationPathAlreadyExistsError:
            # Allow dataset reuse between models.
            pass
        return versioned_path

    def add_versioned_dataset(self, dataset: VersionedDataset) -> str:
        """Adds the versioned dataset to the pool.

        :param dataset: The versioned dataset.
        :return: The dataset's publication path.
        """
        if not hasattr(dataset, "X_train") or not hasattr(dataset, "y_train"):
            raise UnrecognizedTrainingDatasetError
        dataset_publication_path = os.path.join(
            self.datasets_dirname, dataset.name
        )
        republished_dataset_path = os.path.join(
            dataset_publication_path, dataset.version
        )
        try:
            dataset.republish(dataset_publication_path)
        except PublicationPathAlreadyExistsError:
            # Allow dataset reuse between models.
            pass
        return republished_dataset_path

    def remove_model(self, model_name: str) -> None:
        """Removes all matching models, including previous versions.

        :param model_name: The name of the model to remove.
        """
        if model_name not in self:
            raise ArtifactNotInPoolError
        model_path = os.path.join(self.models_dirname, model_name)
        shutil.rmtree(model_path)

    def remove_dataset(self, dataset_name: str) -> None:
        """Removes all matching datasets, including previous versions.

        Note that models using the given dataset will not have valid dataset
        paths after this operation has completed.

        :param dataset_name: The name of the dataset to remove.
        """
        if dataset_name not in self:
            raise ArtifactNotInPoolError
        dataset_path = os.path.join(self.datasets_dirname, dataset_name)
        shutil.rmtree(dataset_path)

    @staticmethod
    def _argmax_version(artifact_versions: list[str]) -> int:
        """Returns the index of the highest (most recent) version.

        :param artifact_versions: The list of versions. Versions are in any PEP
            440 compliant format.
        :return: The index of the highest (most recent) version.
        """
        return max(
            list(range(len(artifact_versions))),
            key=lambda idx: parse_version(artifact_versions[idx]),
        )

    @staticmethod
    def _get_versioned_artifacts(
        base_dir: str,
        latest_only: bool = True,
        filter_names: set[str] | None = None,
    ) -> set[str]:
        """Returns the set of versioned artifacts from the base directory.

        :param base_dir: The directory containing versioned artifacts; the
            model or dataset directory.
        :param latest_only: If True, only return the highest versioned artifact
            of each type. E.g., if there are both mnist_model/v1 and
            mnist_model/v2, return only the path to v2. If False, return all
            paths, regardless of version.
        :param filter_names: If provided, only return paths to artifacts whose
            names are in the set.
        """
        artifact_paths = set()
        artifact_names = os.listdir(base_dir)
        if filter_names:
            artifact_names = [
                artifact_name
                for artifact_name in artifact_names
                if artifact_name in filter_names
            ]
        for artifact_name in artifact_names:
            artifact_path = os.path.join(base_dir, artifact_name)
            artifact_versions = os.listdir(artifact_path)
            if latest_only:
                highest_version_idx = PretrainingPool._argmax_version(
                    artifact_versions
                )
                highest_version_path = os.path.join(
                    artifact_path, artifact_versions[highest_version_idx]
                )
                artifact_paths.add(highest_version_path)
            else:
                for artifact_version in artifact_versions:
                    version_path = os.path.join(
                        artifact_path, artifact_version
                    )
                    artifact_paths.add(version_path)
        return artifact_paths

    def get_available_models(self, latest_only: bool = True) -> set[str]:
        """Returns the set of model paths available in the pool.

        :param latest_only: If True, only return the highest versioned model
            of each type. E.g., if there are both mnist_model/v1 and
            mnist_model/v2, return only the path to v2. If False, return all
            paths, regardless of version.
        :return: The set of model paths available in the pool. All returned
            paths will be VersionedModel paths, and will therefore include
            the version suffix.
        """
        return PretrainingPool._get_versioned_artifacts(
            self.models_dirname, latest_only=latest_only
        )

    def get_available_datasets(self, latest_only: bool = True) -> set[str]:
        """Returns the set of dataset paths available in the pool.

        :param latest_only: If True, only return the highest versioned dataset
            of each type. E.g., if there are both mnist_dataset/v1 and
            mnist_dataset/v2, return only the path to v2. If False, return all
            paths, regardless of version.
        :return: The set of dataset paths available in the pool. All returned
            paths will be VersionedDataset paths, and will therefore include
            the version suffix.
        """
        return PretrainingPool._get_versioned_artifacts(
            self.datasets_dirname, latest_only=latest_only
        )

    def get_model_path(self, model_name: str) -> str:
        """Returns the path to the given model.

        :param model_name: The name of the model in the pool.
        :return: The path to the model; a VersionedModel path.
        """
        # This will return zero or one paths.
        model_paths = PretrainingPool._get_versioned_artifacts(
            self.models_dirname, latest_only=True, filter_names={model_name}
        )
        if not model_paths:
            raise ArtifactNotInPoolError
        return list(model_paths)[0]

    def get_dataset_path(self, dataset_name: str) -> str:
        """Returns the path to the given dataset.

        :param dataset_name: The name of the dataset in the pool.
        :return: The path to the dataset; a VersionedDataset path.
        """
        # This will return zero or one paths.
        dataset_paths = PretrainingPool._get_versioned_artifacts(
            self.datasets_dirname,
            latest_only=True,
            filter_names={dataset_name},
        )
        if not dataset_paths:
            raise ArtifactNotInPoolError
        return list(dataset_paths)[0]

    def clear(self) -> None:
        """Removes all models and datasets from the pool."""
        shutil.rmtree(self.datasets_dirname, ignore_errors=True)
        shutil.rmtree(self.models_dirname, ignore_errors=True)
        Path(self.models_dirname).mkdir(parents=True, exist_ok=True)
        Path(self.datasets_dirname).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _preprocess_dataset_vgg16(
        X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the preprocessed X and y for VGG16.

        :param X: The input features in the same format as the model's
            associated versioned dataset's features.
        :param y: The input labels in the same format as the model's associated
            versioned dataset's labels.
        :return: The preprocessed X and y for VGG16.
        """
        X = smart_resize(X, VGG_INPUT_SHAPE)
        X = preprocess_input_vgg16(X)
        return X, y

    @staticmethod
    def _preprocess_dataset_efficientnetb7(
        X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the preprocessed X and y for EfficientNetB7.

        :param X: The input features in the same format as the model's
            associated versioned dataset's features.
        :param y: The input labels in the same format as the model's associated
            versioned dataset's labels.
        :return: The preprocessed X and y for EfficientNetB7.
        """
        X = smart_resize(X, EFFICIENTNETB7_INPUT_SHAPE)
        X = preprocess_input_effnet(X)
        return X, y

    @staticmethod
    def preprocess_dataset(
        model_name: str, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the preprocessed X and y for the given model.

        Some models require their datasets to be preprocessed because the
        dataset is shared with other models that use a different
        representation. User models will not have their datasets changed.

        :param model_name: The name of the model whose dataset should be
            preprocessed. Valid choices are any of the names in
            OPEN_SOURCE_MODELS.
        :param X: The input features in the same format as the model's
            associated versioned dataset's features.
        :param y: The input labels in the same format as the model's associated
            versioned dataset's labels.
        :return: The preprocessed X and y for the given model.
        """
        if model_name == MODEL_VGG16:
            X, y = PretrainingPool._preprocess_dataset_vgg16(X, y)
        elif model_name == MODEL_EFFICIENTNETB7:
            X, y = PretrainingPool._preprocess_dataset_efficientnetb7(X, y)
        return X, y

    @staticmethod
    def eval_model(
        versioned_model: VersionedModel,
        versioned_dataset: VersionedDataset,
        frac: float = 1.0,
    ) -> float | list[float]:
        """Evaluates the model on the dataset.

        The model and dataset need not be in the pool, but models and datasets
        recognized by the pool will be preprocessed as necessary.

        :param versioned_model: The model to evaluate. This model need not be
            in the pool, but if it is a known model, certain preprocessing
            steps may be applied as necessary.
        :param versioned_dataset: The dataset on which to evaluate the model.
        :param frac: The fraction of the dataset on which to run evaluation.
        :return: The model evaluation results; either a single scalar
            indicating the loss, or a list of scalars indicating the loss and
            subsequent metric values, in order of compilation.
        """
        X_train = versioned_dataset.X_train
        y_train = versioned_dataset.y_train
        samples = int(len(X_train) * frac)
        X_train = X_train[:samples]
        y_train = y_train[:samples]
        X_train, y_train = PretrainingPool.preprocess_dataset(
            versioned_model.name, X_train, y_train
        )
        return versioned_model.model.evaluate(X_train, y_train)

    @staticmethod
    def compile_model(versioned_model: VersionedModel) -> None:
        """Compiles the given model.

        Open source models will be compiled using known-good configurations for
        loss, metrics, etc. This function will raise an error on custom models,
        since compilation procedures applied to them will likely be wrong.

        :param versioned_model: The open source model to compile. Custom models
            will raise an error.
        """
        if versioned_model.name in (MODEL_VGG16, MODEL_EFFICIENTNETB7):
            versioned_model.model.compile(
                loss="categorical_crossentropy",
                metrics=["accuracy", "top_k_categorical_accuracy"],
            )
        else:
            raise PretrainingPoolCannotCompileCustomModelError

    def get_pretrained_model(
        self, model: Model, X_train: np.ndarray, y_train: np.ndarray
    ) -> Model:
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
