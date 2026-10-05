import warnings
import pytest


@pytest.fixture(autouse=True)
def suppress_external_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
