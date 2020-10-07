import pytest

from mikeio.res1d import read, Res1D


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
