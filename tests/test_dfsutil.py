from pydhi.dfs_util import type_list, unit_list


def test_type_list_finds_some_types():

    types = type_list('Water level')

    assert len(types) > 0


def test_unit_list_finds_some_types():

    units = unit_list(100000)

    assert len(units) > 0
