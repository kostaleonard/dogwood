"""Contains custom errors."""


class NotADenseLayerError(ValueError):
    """Raised when a function expects a dense layer as input, but the user
    supplies something else."""
