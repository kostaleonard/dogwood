"""Conducts experiments with model-based transfer techniques."""

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


if __name__ == '__main__':
    main()
