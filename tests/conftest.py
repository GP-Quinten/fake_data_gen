"""This file contains all the variables that you can use in the different tests."""

import pytest


@pytest.fixture()
def text_variable():
    return "This is only an example."
