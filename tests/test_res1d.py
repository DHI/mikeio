import pytest
from mikeio.res1d import read, Res1D, QueryData


def test_query_validate():
    # Good variable types
    query = QueryData("WaterLevel")
    query = QueryData("Discharge")
    query = QueryData("Pollutant")

    # Bad variable type
    with pytest.raises(TypeError):
        QueryData(666)
    
    # Bad string variable type
    with pytest.raises(ValueError):
        QueryData("BadVariableType")
    
    # Bad branch type
    with pytest.raises(TypeError):
        QueryData("WaterLevel", BranchName=666)
    
    # Bad chainage type
    with pytest.raises(TypeError):
        QueryData("WaterLevel", "branch", Chainage="BadChainage")
    
    # Cannot set a chainage with no branch
    with pytest.raises(ValueError):
        QueryData("WaterLevel", None, 10)


def test_query_repr():
    query = QueryData("WaterLevel", "104l1", 34.4131)
    expected = ("QueryData(VariableType='WaterLevel', BranchName='104l1', "
                "Chainage=34.4131)")
    assert repr(query) == expected


def test_query_iter():
    query = QueryData("WaterLevel", "104l1", 34.4131)
    vt, bn, c = query
    assert vt == "WaterLevel"
    assert bn == "104l1"
    assert c == 34.4131

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
