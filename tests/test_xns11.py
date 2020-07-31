import pytest

from mikeio.xns11 import (
    read, Xns11, QueryData, FileNotOpenedError, DataNotFoundInFile
)


def test_query_validate():
    # Bad topo id type
    with pytest.raises(TypeError):
        QueryData(666)

    # Bad reach type
    with pytest.raises(TypeError):
        QueryData("topoid1", reach_name=666)

    # Bad chainage type
    with pytest.raises(TypeError):
        QueryData("topoid1", "reach1", chainage="BadChainage")
    
    # Cannot set a chainage with no reach
    with pytest.raises(ValueError):
        QueryData("topoid1", None, 10)


def test_query_repr():
    query = QueryData("topoid1", "reach1", 34.4131)
    expected = ("QueryData(topoid_name='topoid1', reach_name='reach1', "
                "chainage=34.4131)")
    assert repr(query) == expected


@pytest.fixture
def file():
    return "tests/testdata/xsections.xns11"


def test_file_does_not_exist():
    file = "tests/testdata/not_a_file.xns11"

    query = QueryData("baseline")

    with pytest.raises(FileExistsError):
        assert read(file, [query])


def test_get_properties_if_not_opened(file):
    """Public properties cannot be accessed if the file is not opened"""
    r = Xns11(file)
    r.close()

    with pytest.raises(FileNotOpenedError) as excinfo:
        r.topoid_names
    assert "topoid_names" in str(excinfo.value)

    with pytest.raises(FileNotOpenedError) as excinfo:
        r.reach_names
    assert "reach_names" in str(excinfo.value)


@pytest.mark.parametrize("query,expected_bottom", [
    (QueryData("topoid1", "reach1", 58.68), 1626.16),
    (QueryData("topoid2", "reach2", -50), 1611.42),
    (QueryData("topoid1", "reach3", 11150.42), 1616.038),
    (QueryData("topoid1", "reach4", 13645.41), 1594.923)
])
def test_read_single_query_as_list(file, query, expected_bottom):
    geometry = read(file, [query])
    assert pytest.approx(round(geometry[geometry.columns[1]].min(), 3)) == expected_bottom


@pytest.mark.parametrize("query,expected_bottom", [
    (QueryData("topoid1", "reach1", 58.68), 1626.16),
    (QueryData("topoid2", "reach2", -50), 1611.42),
    (QueryData("topoid1", "reach3", 11150.42), 1616.038),
    (QueryData("topoid1", "reach4", 13645.41), 1594.923)
])
def test_read_single_query(file, query, expected_bottom):
    geometry = read(file, query)
    assert pytest.approx(round(geometry[geometry.columns[1]].min(), 3)) == expected_bottom
 

def test_read_bad_queries(file):
    """Querying data not available in the file must return an error"""

    # Bad topo-id
    with pytest.raises(DataNotFoundInFile) as excinfo:
        read(file, [QueryData("bad_topoid")])
    assert "bad_topoid" in str(excinfo.value)
 
    # Bad reach name
    with pytest.raises(DataNotFoundInFile) as excinfo:
        read(file, [QueryData("topoid1", "bad_reach_name")])
    assert "bad_reach_name" in str(excinfo.value)

    # Bad chainage
    with pytest.raises(DataNotFoundInFile) as excinfo:
        read(file, [QueryData("topoid1", "reach1", 666)])
    assert "666" in str(excinfo.value)


def test_read_multiple_queries(file):
    q1 = QueryData("topoid1", "reach1", 58.68)
    q2 = QueryData("topoid2", "reach2", -50)
    geometry = read(file, [q1, q2])
    geometry_min = [666, 666]
    geometry_min[0] = geometry[geometry.columns[1]].min()
    geometry_min[1] = geometry[geometry.columns[3]].min()
    assert pytest.approx(round(geometry_min[0], 3)) == 1626.16
    assert pytest.approx(round(geometry_min[1], 3)) == 1611.42


def test_read_reach(file):
    q_reach = QueryData("topoid2", "reach2")
    geometry = read(file, [q_reach])
    assert list(geometry.columns) == [
        'x topoid2 reach2 -50.0',
        'z topoid2 reach2 -50.0',
        'x topoid2 reach2 64.376',
        'z topoid2 reach2 64.376',
        'x topoid2 reach2 135.0',
        'z topoid2 reach2 135.0',
    ]


def test_read_multiple_reaches(file):
    q_reach1 = QueryData("topoid1", "reach3")
    q_reach4 = QueryData("topoid1", "reach4")
    geometry = read(file, [q_reach1, q_reach4])
    assert len(geometry.columns) == 104


def test_read_all_topoid(file):
    q_topoid1 = QueryData("topoid1")
    geometry = read(file, [q_topoid1])
    assert len(geometry.columns) == 116
    assert geometry[geometry.columns[0]].count() == 37

    q_topoid2 = QueryData("topoid2")
    geometry = read(file, [q_topoid2])
    assert len(geometry.columns) == 6
    assert geometry[geometry.columns[0]].count() == 4
