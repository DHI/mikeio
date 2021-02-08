import pathlib

import mikeio
import pytest

_testdata_directory = pathlib.Path(__file__).parent.resolve() / "testdata"


@pytest.fixture(scope="session")
def testdata_directory() -> pathlib.Path:
    return _testdata_directory


@pytest.fixture(scope="function")
def dfsu_hd2d(testdata_directory) -> mikeio.Dfsu:
    return mikeio.Dfsu(filename=str(testdata_directory / "HD2D.dfsu"))
