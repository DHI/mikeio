import pytest

from mikeio.res1d import read, Res1D, mike1d_quantities, to_numpy
import numpy as np


@pytest.fixture
def test_file_path():
    return "tests/testdata/Exam6Base.res1d"


@pytest.fixture
def test_file(test_file_path):
    return Res1D(test_file_path)


def test_file_does_not_exist():
    with pytest.raises(FileExistsError):
        assert read("tests/testdata/not_a_file.res1d")


def test_read(test_file_path):
    ts = read(test_file_path)
    assert len(ts) == 110


def test_mike1d_quantities():
    quantities = mike1d_quantities()
    assert "WaterLevel" in quantities


def test_quantities(test_file):
    quantities = test_file.quantities
    assert len(quantities) == 2


@pytest.mark.parametrize("quantity,id,chainage,expected_max", [
    ("WaterLevel", "104l1", 34.4131, 197.046),
    ("WaterLevel", "9l1", 10, 195.165),
    ("Discharge", "100l1", 23.8414, 0.1),
    ("Discharge", "9l1", 5, 0.761)
])
def test_read_reach(test_file, quantity, id, chainage, expected_max):
    data = test_file.query.GetReachValues(id, chainage, quantity)
    data = to_numpy(data)
    actual_max = round(np.max(data), 3)
    assert pytest.approx(actual_max) == expected_max


@pytest.mark.parametrize("quantity,id,expected_max", [
    ("WaterLevel", "1", 195.669),
    ("WaterLevel", "2", 195.823)
])
def test_read_node(test_file, quantity, id, expected_max):
    data = test_file.query.GetNodeValues(id, quantity)
    data = to_numpy(data)
    actual_max = round(np.max(data), 3)
    assert pytest.approx(actual_max) == expected_max


def test_time_index(test_file):
    assert len(test_file.time_index) == 110


def test_start_time(test_file):
    assert test_file.start_time == test_file.time_index.min()