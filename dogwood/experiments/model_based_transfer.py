"""Conducts experiments with model-based transfer techniques."""

from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.versioned_model import VersionedModel
from dogwood.pretraining.pretraining_pool import PretrainingPool


def get_imagenet_accuracy(
        versioned_dataset: VersionedDataset,
        versioned_model: VersionedModel,
        top_n: int = 1) -> float:
    """TODO"""
    return versioned_model.model(versioned_dataset.X_train)
    return versioned_model.model.evaluate(
        versioned_dataset.X_train,
        versioned_dataset.y_train
    )


def main() -> None:
    """Runs the program."""
    pool = PretrainingPool()
    print('Models:')
    for model_path in pool.get_available_models():
        print(model_path)
    print()
    print('Datasets:')
    for dataset_path in pool.get_available_datasets():
        print(dataset_path)
    for model_path in pool.get_available_models():
        versioned_model = VersionedModel(model_path)
        versioned_dataset = VersionedDataset(versioned_model.dataset_path)
        print(versioned_model.name)
        print(versioned_dataset.name)
        #print(get_imagenet_accuracy(versioned_dataset, versioned_model))


if __name__ == '__main__':
    main()
