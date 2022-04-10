"""Tests model_based_transfer.py."""

import os
import pytest
from dogwood.pretraining.pretraining_pool import PRETRAINED_DIRNAME
from dogwood.experiments.model_based_transfer import main


@pytest.mark.slowtest
def test_main_creates_pretraining_pool() -> None:
    """Tests that main creates a pretraining pool."""
    # We will not delete the existing pretraining directory because that would
    # be very disruptive to users.
    main()
    assert os.path.exists(PRETRAINED_DIRNAME)
