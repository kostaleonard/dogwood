"""Contains the ImageNetDataProcessor class.

The ImageNet class index is taken from TensorFlow. Many pretrained models use
this ordering for ImageNet classes.

The expected directory structure for the dataset is based on this reference:
https://www.kaggle.com/ifigotin/imagenetmini-1000
The Kaggle page links to a pytorch ImageNet preprocessing script that uses the
same format, so this implementation will be applicable in many cases.
"""
# pylint: disable=no-name-in-module

import os
import json
import numpy as np
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from mlops.dataset.invertible_data_processor import InvertibleDataProcessor

IMAGE_SCALE_HEIGHT = 256
IMAGE_SCALE_WIDTH = 256
IMAGE_TARGET_SIZE = (IMAGE_SCALE_HEIGHT, IMAGE_SCALE_WIDTH)
CLASS_INDEX_PATH = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "data/imagenet_class_index.json"
)
CLASS_INDEX_HASH = "c2c37ea517e94d9795004a39431a14cb"


class ImageNetDataProcessor(InvertibleDataProcessor):
    """Transforms the ImageNet data into tensors."""

    def __init__(self) -> None:
        """Instantiates the object."""
        self.class_index = ImageNetDataProcessor.get_class_index()
        self.designator_index = ImageNetDataProcessor.get_designator_index(
            self.class_index
        )

    @staticmethod
    def get_class_index() -> dict[str, list[str]]:
        """Returns the ImageNet class index.

        :return: The ImageNet class index; a dictionary whose keys are the
            class indices as strings (i.e., '0', '1', ..., '999'), and whose
            values are lists of two entries: the class designator (e.g.,
            'n01440764') and the class name (e.g., 'tench').
        """
        file_path = get_file(
            "imagenet_class_index.json",
            CLASS_INDEX_PATH,
            cache_subdir="models",
            file_hash=CLASS_INDEX_HASH,
        )
        with open(file_path, "r", encoding="utf-8") as infile:
            return json.loads(infile.read())

    @staticmethod
    def get_designator_index(
        class_index: dict[str, list[str]]
    ) -> dict[str, str]:
        """Returns the ImageNet designator index.

        :param class_index: The ImageNet class index.
        :return: The ImageNet designator index; a dictionary whose keys are the
            class designators (e.g., 'n01440764') and whose values are the
            class indices as strings (i.e., '0', '1', ..., '999').
        """
        return {
            designator_and_name[0]: idx
            for idx, designator_and_name in class_index.items()
        }

    @staticmethod
    def _get_raw_features_and_labels_subset(
        subset_path: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the raw features and labels from the subset path.

        :param subset_path: The path to the train/val/test subset.
        :return: A 2-tuple of the features and labels tensors.
        """
        subset_features = []
        subset_labels = []
        class_dirnames = os.listdir(subset_path)
        for class_name in class_dirnames:
            class_images_path = os.path.join(subset_path, class_name)
            image_names = os.listdir(class_images_path)
            for image_name in image_names:
                image_path = os.path.join(class_images_path, image_name)
                image = load_img(image_path, target_size=IMAGE_TARGET_SIZE)
                tensor = img_to_array(image, dtype=np.uint8)
                subset_features.append(tensor)
                subset_labels.append(class_name)
        subset_features = np.array(subset_features, dtype=np.uint8)
        subset_labels = np.array(subset_labels)
        return subset_features, subset_labels

    def get_raw_features_and_labels(
        self, dataset_path: str
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Returns the raw feature and label tensors from the dataset path.
        This method is specifically used for the train/val/test sets and not
        input data for prediction, because in some cases the features and
        labels need to be read simultaneously to ensure proper ordering of
        features and labels.

        Raw features are tensors of shape m x h x w x c, where m is the number
        of images, h is the image height, w is the image width, and c is the
        number of channels (3 for RGB), with all values in the interval
        [0, 255]. Raw labels are tensors of shape m, where m is the number of
        images. Each element is the string class designator (e.g.,
        'n01440764').

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset, specifically
            train/val/test and not prediction data.
        :return: A 2-tuple of the features dictionary and labels dictionary,
            with matching keys and ordered tensors.
        """
        features = {}
        labels = {}
        for subset in "train", "val", "test":
            subset_path = os.path.join(dataset_path, subset)
            if not os.path.exists(subset_path):
                continue
            (
                subset_features,
                subset_labels,
            ) = ImageNetDataProcessor._get_raw_features_and_labels_subset(
                subset_path
            )
            features_key = f"X_{subset}"
            labels_key = f"y_{subset}"
            features[features_key] = subset_features
            labels[labels_key] = subset_labels
        return features, labels

    def get_raw_features(self, dataset_path: str) -> dict[str, np.ndarray]:
        """Returns the raw feature tensors from the prediction dataset path.
        Raw features are tensors of shape m x h x w x c, where m is the number
        of images, h is the image height, w is the image width, and c is the
        number of channels (3 for RGB), with all values in the interval
        [0, 255]. The data type is uint8.

        Images used for prediction should be in the same directory structure as
        train/val/test images, namely that images should be in a subdirectory
        under the name 'pred'. The name of the subdirectory(-ies) need not be
        the class name, since in general that is not known beforehand.

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset.
        :return: A dictionary whose values are feature tensors and whose
            corresponding keys are the names by which those tensors should be
            referenced. The returned keys will be a subset of {'X_train',
            'X_val', 'X_test', 'X_pred'}.
        """
        features = {}
        for subset in "train", "val", "test", "pred":
            subset_path = os.path.join(dataset_path, subset)
            if not os.path.exists(subset_path):
                continue
            (
                subset_features,
                _,
            ) = ImageNetDataProcessor._get_raw_features_and_labels_subset(
                subset_path
            )
            features_key = f"X_{subset}"
            features[features_key] = subset_features
        return features

    def preprocess_features(
        self, raw_feature_tensor: np.ndarray
    ) -> np.ndarray:
        """Returns the preprocessed feature tensor from the raw tensor. The
        preprocessed features are how training/validation/test as well as
        prediction data are fed into downstream models. The preprocessed
        tensors are of shape m x h x w x c, where m is the number of images, h
        is the image height, w is the image width, and c is the number of
        channels (3 for RGB), with all values in the interval [0, 255].

        Because each model may preprocesses ImageNet images in a unique way,
        the raw features are returned.

        :param raw_feature_tensor: The raw features to be preprocessed.
        :return: The preprocessed feature tensor. This tensor is ready for
            downstream model consumption.
        """
        # No preprocessing is applied.
        return raw_feature_tensor.copy()

    def preprocess_labels(self, raw_label_tensor: np.ndarray) -> np.ndarray:
        """Returns the preprocessed label tensor from the raw tensor. The
        preprocessed labels are how training/validation/test as well as
        prediction data are fed into downstream models.

        :param raw_label_tensor: The raw labels to be preprocessed.
        :return: The preprocessed label tensor. This tensor is ready for
            downstream model consumption.
        """
        preprocessed = np.zeros((len(raw_label_tensor), len(self.class_index)))
        for example_idx, designator in enumerate(raw_label_tensor):
            class_idx = int(self.designator_index[designator])
            preprocessed[example_idx, class_idx] = 1
        return preprocessed

    def unpreprocess_features(self, feature_tensor: np.ndarray) -> np.ndarray:
        """Returns the raw feature tensor from the preprocessed tensor; inverts
        preprocessing. Improves model interpretability by enabling users to
        transform model inputs into real-world values.

        :param feature_tensor: The preprocessed features to be inverted.
        :return: The raw feature tensor.
        """
        # No preprocessing was applied.
        return feature_tensor.copy()

    def unpreprocess_labels(self, label_tensor: np.ndarray) -> np.ndarray:
        """Returns the raw label tensor from the preprocessed tensor; inverts
        preprocessing. Improves model interpretability by enabling users to
        transform model outputs into real-world values.

        :param label_tensor: The preprocessed labels to be inverted.
        :return: The raw label tensor.
        """
        unpreprocessed = []
        class_indices = np.argmax(label_tensor, axis=1)
        for idx in class_indices:
            designator = self.class_index[str(idx)][0]
            unpreprocessed.append(designator)
        return np.array(unpreprocessed)
