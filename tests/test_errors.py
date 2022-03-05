"""Tests errors.py."""

from dogwood.errors import NotADenseLayerError, InvalidExpansionStrategyError


def test_not_a_dense_layer_error_extends_value_error() -> None:
    """Tests that NotADenseLayerError extends ValueError."""
    err = NotADenseLayerError()
    assert isinstance(err, ValueError)


def test_invalid_expansion_strategy_error_extends_value_error() -> None:
    """Tests that InvalidExpansionStrategyError extends ValueError."""
    err = InvalidExpansionStrategyError()
    assert isinstance(err, ValueError)
