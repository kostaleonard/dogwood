"""Tests errors.py."""

from dogwood.errors import NotADenseLayerError


def test_not_a_dense_layer_error_extends_value_error() -> None:
    """Tests that NotADenseLayerError extends ValueError."""
    err = NotADenseLayerError()
    assert isinstance(err, ValueError)
