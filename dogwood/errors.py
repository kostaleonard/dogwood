"""Contains custom errors."""


class NotADenseLayerError(ValueError):
    """Raised when a function expects a dense layer as input, but the user
    supplies something else."""


class InvalidExpansionStrategyError(ValueError):
    """Raised when a user attempts to expand a model, but the strategy for
    setting new weights is not one of the valid options."""
