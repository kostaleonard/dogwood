"""Contains custom errors."""


class NotADenseLayerError(ValueError):
    """Raised when a function expects a dense layer as input, but the user
    supplies something else."""


class InvalidExpansionStrategyError(ValueError):
    """Raised when a user attempts to expand a model, but the strategy for
    setting new weights is not one of the valid options."""


class PretrainingPoolAlreadyContainsModelError(FileExistsError):
    """Raised when a user adds two models with the same name to the pretraining
    pool."""


class NoSuchOpenSourceModelError(ValueError):
    """Raised when a user attempts to instantiate a pretraining pool with an
    unknown or unsupported open source model."""


class UnrecognizedTrainingDatasetError(ValueError):
    """Raised when a user adds a dataset with no training feature/label tensors
    to a pretraining pool."""


class PretrainingPoolCannotCompileCustomModelError(ValueError):
    """Raised when a user attempts to call PretrainingPool.compile_model on a
    custom model."""


class ArtifactNotInPoolError(ValueError):
    """Raised when a user attempts to retrieve the path to a model or dataset
    that is not in the PretrainingPool."""
