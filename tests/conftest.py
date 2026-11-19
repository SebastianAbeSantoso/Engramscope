import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("ENGRAMSCOPE_TEST_NETWORK") == "1":
        return
    skip = pytest.mark.skip(reason="needs the live model files; set ENGRAMSCOPE_TEST_NETWORK=1")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def model():
    from engramscope.readout import Model

    return Model()
