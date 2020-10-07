import pytest

from mikeio.res1d import read, Res1D
import numpy as np


@pytest.fixture
def test_file_path():
    return "tests/testdata/Exam6Base.res1d"


def test_file_does_not_exist():
    with pytest.raises(FileExistsError):
        assert read("tests/testdata/not_a_file.res1d")


def test_read(test_file_path):
    ts = read(test_file_path)
    assert len(ts) == 110


def test_quantities(test_file_path):
    res1d = Res1D(test_file_path)
    quantities = res1d.quantities
    assert len(quantities) == 2


# @pytest.mark.parametrize("query,expected_max", [
#    (QueryData("WaterLevel", "104l1", 34.4131), 197.046),
#    (QueryData("WaterLevel", "9l1", 10), 195.165),
#    (QueryData("Discharge", "100l1", 23.8414), 0.1),
#    (QueryData("Discharge", "9l1", 5), 0.761)
# ])
def test_read_reach(test_file_path):
    res1d = Res1D(test_file_path)
    data = res1d.query.GetReachValues("104l1", 34.4131, "WaterLevel")
    data = np.fromiter(data, np.float64)
    expected_max = 197.046

    assert pytest.approx(round(np.max(data), 3)) == expected_max
