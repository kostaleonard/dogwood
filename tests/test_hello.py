"""Placeholder test."""

from dogwood.hello import hello, HELLO_STR


def test_hello() -> None:
    """Tests hello."""
    assert hello() == HELLO_STR
