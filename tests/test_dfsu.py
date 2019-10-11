from pydhi.dfsu import dfsu


def test_read_all_items_returns_all_items_and_names():
    filename = r"tests/testdata/HD2D.dfsu"
    dfs = dfsu()

    (data, t, names) = dfs.read(filename)

    assert len(data) == 4
    assert len(names) == 4


def test_read_single_item_returns_single_item():
    filename = r"tests/testdata/HD2D.dfsu"
    dfs = dfsu()

    (data, t, names) = dfs.read(filename, item_numbers=[3])

    assert len(data) == 1
    assert len(names) == 1


def test_read_selected_item_returns_correct_items():
    filename = r"tests/testdata/HD2D.dfsu"
    dfs = dfsu()

    (data, t, names) = dfs.read(filename, item_numbers=[0,3])

    assert len(data) == 2
    assert len(names) == 2
    assert names[0] == "Surface elevation"
    assert names[1] == "Current speed"

