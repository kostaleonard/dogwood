"""Conducts experiments with model-based transfer techniques."""

from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.versioned_model import VersionedModel
from dogwood.pretraining.pretraining_pool import PretrainingPool


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
        PretrainingPool.compile_model(versioned_model)
        print(versioned_model.name)
        print(versioned_dataset.name)
        metrics = PretrainingPool.eval_model(
            versioned_model, versioned_dataset, frac=0.01)
        print(metrics)


if __name__ == '__main__':
    main()
