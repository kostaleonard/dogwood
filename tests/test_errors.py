"""Tests errors.py."""

from dogwood.errors import NotADenseLayerError, \
    InvalidExpansionStrategyError, PretrainingPoolAlreadyContainsModelError, \
    NoSuchOpenSourceModelError, UnrecognizedTrainingDatasetError, \
    PretrainingPoolCannotCompileCustomModelError, ArtifactNotInPoolError


def test_not_a_dense_layer_error_extends_value_error() -> None:
    """Tests that NotADenseLayerError extends ValueError."""
    err = NotADenseLayerError()
    assert isinstance(err, ValueError)


def test_invalid_expansion_strategy_error_extends_value_error() -> None:
    """Tests that InvalidExpansionStrategyError extends ValueError."""
    err = InvalidExpansionStrategyError()
    assert isinstance(err, ValueError)


def test_pool_contains_model_error_extends_file_exists_error() -> None:
    """Tests that PretrainingPoolAlreadyContainsModelError extends
    FileExistsError."""
    err = PretrainingPoolAlreadyContainsModelError()
    assert isinstance(err, FileExistsError)


def test_no_such_open_source_model_error_extends_value_error() -> None:
    """Tests that NoSuchOpenSourceModelError extends ValueError."""
    err = NoSuchOpenSourceModelError()
    assert isinstance(err, ValueError)


def test_unrecognized_training_dataset_error_extends_value_error() -> None:
    """Tests that UnrecognizedTrainingDatasetError extends ValueError."""
    err = UnrecognizedTrainingDatasetError()
    assert isinstance(err, ValueError)


def test_cannot_compile_model_error_extends_value_error() -> None:
    """Tests that PretrainingPoolCannotCompileCustomModelError extends
    ValueError."""
    err = PretrainingPoolCannotCompileCustomModelError()
    assert isinstance(err, ValueError)


def test_artifact_not_in_pool_error_extends_value_error() -> None:
    """Tests that ArtifactNotInPoolError extends
    ValueError."""
    err = ArtifactNotInPoolError()
    assert isinstance(err, ValueError)
