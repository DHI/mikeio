from pydhi import dfs_util


def test_type_list_finds_some_types():

    util = dfs_util.dfs_util()

    types = util.type_list('Water level')

    assert len(types) > 0

def test_unit_list_finds_some_types():

    util = dfs_util.dfs_util()

    units = util.unit_list(100000)

    assert len(units) > 0
