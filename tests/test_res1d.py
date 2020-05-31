import pytest

from mikeio.res1d import (
    read, QueryData, Res1D, FileNotOpenedError, DataNotFoundInFile
)


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
        QueryData("WaterLevel", branch_name=666)

    # Bad chainage type
    with pytest.raises(TypeError):
        QueryData("WaterLevel", "branch", chainage="BadChainage")

    # Cannot set a chainage with no branch
    with pytest.raises(ValueError):
        QueryData("WaterLevel", None, 10)


def test_query_repr():
    query = QueryData("WaterLevel", "104l1", 34.4131)
    expected = ("QueryData(variable_type='WaterLevel', branch_name='104l1', "
                "chainage=34.4131)")
    assert repr(query) == expected


def test_query_iter():
    query = QueryData("WaterLevel", "104l1", 34.4131)
    vt, bn, c = query
    assert vt == "WaterLevel"
    assert bn == "104l1"
    assert c == 34.4131


@pytest.fixture
def file():
    return "tests/testdata/Exam6Base.res1d"


def test_file_does_not_exist():
    file = "tests/testdata/not_a_file.res1d"

    query = QueryData("WaterLevel")

    with pytest.raises(FileExistsError):
        assert read(file, [query])


def test_get_properties_if_not_opened(file):
    """Public properties cannot be accessed if the file is not opened"""
    r = Res1D(file)
    r.close()

    with pytest.raises(FileNotOpenedError) as excinfo:
        r.data_types
    assert "data_types" in str(excinfo.value)

    with pytest.raises(FileNotOpenedError) as excinfo:
        r.reach_names
    assert "reach_names" in str(excinfo.value)
    
    with pytest.raises(FileNotOpenedError) as excinfo:
        r.time_index
    assert "time_index" in str(excinfo.value)


@pytest.mark.parametrize("query,expected_max", [
    (QueryData("WaterLevel", "104l1", 34.4131), 197.046),
    (QueryData("WaterLevel", "9l1", 10), 195.165),
    (QueryData("Discharge", "100l1", 23.8414), 0.1),
    (QueryData("Discharge", "9l1", 5), 0.761)
])
def test_read_single_query_as_list(file, query, expected_max):
    ts = read(file, [query])
    assert len(ts) == 110
    assert pytest.approx(round(ts.max()[0], 3)) == expected_max


@pytest.mark.parametrize("query,expected_max", [
    (QueryData("WaterLevel", "104l1", 34.4131), 197.046),
    (QueryData("WaterLevel", "9l1", 10), 195.165),
    (QueryData("Discharge", "100l1", 23.8414), 0.1),
    (QueryData("Discharge", "9l1", 5), 0.761)
])
def test_read_single_query(file, query, expected_max):
    ts = read(file, query)
    assert len(ts) == 110
    assert pytest.approx(round(ts.max()[0], 3)) == expected_max


def test_read_bad_queries(file):
    """Querying data not available in the file must return an error"""

    # Bad variable type
    with pytest.raises(DataNotFoundInFile) as excinfo:
        read(file, [QueryData("Pollutant")])
    assert "Pollutant" in str(excinfo.value)
 
    # Bad branch name
    with pytest.raises(DataNotFoundInFile) as excinfo:
        read(file, [QueryData("WaterLevel", "bad_branch_name")])
    assert "bad_branch_name" in str(excinfo.value)

    # Bad chainage
    with pytest.raises(DataNotFoundInFile) as excinfo:
        read(file, [QueryData("WaterLevel", "104l1", 666)])
    assert "666" in str(excinfo.value)


def test_read_multiple_queries(file):
    q1 = QueryData("WaterLevel", "104l1", 34.4131)
    q2 = QueryData("Discharge", "9l1", 5)
    ts = read(file, [q1, q2])
    assert ts.shape == (110, 2)
    ts_max = ts.max()
    assert pytest.approx(round(ts_max[0], 3)) == 197.046
    assert pytest.approx(round(ts_max[1], 3)) == 0.761


def test_read_reach(file):
    q_reach = QueryData("WaterLevel", "118l1")
    ts = read(file, [q_reach])
    assert ts.shape == (110, 3)
    assert list(ts.columns) == ['WaterLevel 118l1 0.0', 'WaterLevel 118l1 49.443',
                                'WaterLevel 118l1 98.887']


def test_read_multiple_reaches(file):
    q_reach1 = QueryData("WaterLevel", "118l1")
    q_reach2 = QueryData("Discharge", "113l1")
    ts = read(file, [q_reach1, q_reach2])
    assert ts.shape == (110, 5)


def test_read_all_reaches(file):
    q_waterlevel = QueryData("WaterLevel")
    ts = read(file, [q_waterlevel])
    # Note that it includes 4 water level structure points
    assert ts.shape == (110, 247)

    q_discharge = QueryData("Discharge")
    ts = read(file, [q_discharge])
    # Note that it includes 2 discharge structure points
    assert ts.shape == (110, 129)
