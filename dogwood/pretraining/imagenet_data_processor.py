"""Contains the ImageNetDataProcessor class.

The ImageNet class index is taken from TensorFlow. Many pretrained models use
this ordering for ImageNet classes.

The expected directory structure for the dataset is based on this reference:
https://www.kaggle.com/ifigotin/imagenetmini-1000
The Kaggle page links to a pytorch ImageNet preprocessing script that uses the
same format, so this implementation will be applicable in many cases.
"""

import json
import numpy as np
from tensorflow.keras.utils import get_file
from mlops.dataset.invertible_data_processor import InvertibleDataProcessor

IMAGE_SCALE_HEIGHT = 256
IMAGE_SCALE_WIDTH = 256
CLASS_INDEX_PATH = ('https://storage.googleapis.com/download.tensorflow.org/'
                    'data/imagenet_class_index.json')
CLASS_INDEX_HASH = 'c2c37ea517e94d9795004a39431a14cb'


class ImageNetDataProcessor(InvertibleDataProcessor):
    """Transforms the ImageNet data into tensors."""

    def __init__(self) -> None:
        """Instantiates the object."""
        self.class_index = ImageNetDataProcessor.get_class_index()

    @staticmethod
    def get_class_index() -> dict[str, list[str]]:
        """Returns the ImageNet class index.

        :return: The ImageNet class index; a dictionary whose keys are the
            class indices as strings (i.e., '0', '1', ..., '999'), and whose
            values are lists of two entries: the class designator (e.g.,
            'n01440764') and the class name (e.g., 'tench').
        """
        file_path = get_file(
            'imagenet_class_index.json',
            CLASS_INDEX_PATH,
            cache_subdir='models',
            file_hash=CLASS_INDEX_HASH)
        with open(file_path, 'r', encoding='utf-8') as infile:
            return json.loads(infile.read())

    def get_raw_features_and_labels(self, dataset_path: str) -> \
            tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
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
        # TODO

    def get_raw_features(self, dataset_path: str) -> dict[str, np.ndarray]:
        """Returns the raw feature tensors from the prediction dataset path.
        Raw features are tensors of shape m x h x w x c, where m is the number
        of images, h is the image height, w is the image width, and c is the
        number of channels (3 for RGB), with all values in the interval
        [0, 255]. The data type is uint8.

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset.
        :return: A dictionary whose values are feature tensors and whose
            corresponding keys are the names by which those tensors should be
            referenced. The returned keys will be {'X_train', 'X_val',
            'X_test'} if the directory indicated by dataset_path ends with
            'trainvaltest', and {'X_pred'} otherwise.
        """
        # TODO

    def preprocess_features(
            self, raw_feature_tensor: np.ndarray) -> np.ndarray:
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
        # TODO

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
        # TODO
