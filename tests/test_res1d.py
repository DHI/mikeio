import pytest
from mikeio.res1d import Res1D, ExtractionPoint


def get_test_query():
    query = ExtractionPoint()
    query.BranchName = "104l1"
    query.Chainage = 34.4131
    query.VariableType = "WaterLevel"

    return query


def test_file_does_not_exist():
    file = "tests/testdata/not_a_file.res1d"

    query = get_test_query()

    with pytest.raises(FileExistsError):
        assert Res1D().read(file, [query])


def test_read_single_item():
    file = "tests/testdata/Exam6Base.res1d"
    query = get_test_query()
    ts = Res1D().read(file, [query])

    assert len(ts) == 110
