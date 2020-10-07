import pytest

from mikeio.res1d import (
    read, Res1D
)


@pytest.fixture
def file():
    return "tests/testdata/Exam6Base.res1d"


def test_file_does_not_exist():
    file = "tests/testdata/not_a_file.res1d"

    with pytest.raises(FileExistsError):
        assert read(file)


def test_read_single_query(file):
    ts = read(file)
    assert len(ts) == 110

    res1d = Res1D(file)
    q = res1d.quantities
