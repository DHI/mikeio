import pytest

from mikeio.xns11 import Xns11, QueryData


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
    expected = ("QueryData(topo_id='topoid1', reach_name='reach1', "
                "chainage=34.4131)")
    assert repr(query) == expected


def test_read():
    file_path = "tests/testdata/x_sections.xns11"
    query = QueryData('baseline', 'BigCreek', 741.71)
    geometry = Xns11().read(file_path, [query])

    assert geometry is not None
