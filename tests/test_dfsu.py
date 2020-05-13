import os
from shutil import copyfile
import numpy as np
import pytest

from mikeio import Dfsu, Mesh
from mikeio.eum import ItemInfo


def test_read_all_items_returns_all_items_and_names():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()

    (data, t, items) = dfs.read(filename)

    assert len(data) == 4
    assert len(items) == 4


def test_write(tmpdir):

    infilename = os.path.join("tests", "testdata", "HD2D.dfsu")
    outfilename = os.path.join(tmpdir.dirname, "adjusted.dfsu")

    copyfile(infilename, outfilename)
    dfs = Dfsu()

    (data, t, items) = dfs.read(infilename)

    # Do arbitrary calculation
    data[0] = data[0] * 2.0

    dfs.write(outfilename, data)


def test_read_single_item_returns_single_item():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()

    (data, t, items) = dfs.read(filename, item_numbers=[3])

    assert len(data) == 1
    assert len(items) == 1


def test_read_returns_array_time_dimension_first():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()

    (data, t, items) = dfs.read(filename, item_numbers=[3])

    assert data[0].shape == (9, 884)


def test_read_selected_item_returns_correct_items():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()

    (data, t, items) = dfs.read(filename, item_numbers=[0, 3])

    assert len(data) == 2
    assert len(items) == 2
    assert items[0].name == "Surface elevation"
    assert items[1].name == "Current speed"


def test_read_selected_item_names_returns_correct_items():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()

    (data, t, items) = dfs.read(
        filename, item_names=["Surface elevation", "Current speed"]
    )

    assert len(data) == 2
    assert len(items) == 2
    assert items[0].name == "Surface elevation"
    assert items[1].name == "Current speed"


def test_read_all_time_steps():

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()

    ds = dfs.read(filename, item_numbers=[0, 3])

    assert len(ds.time) == 9
    assert ds.data[0].shape[0] == 9


def test_read_single_time_step():

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()

    ds = dfs.read(filename, item_numbers=[0, 3], time_steps=[1])

    assert len(ds.time) == 1
    assert ds.data[0].shape[0] == 1


def test_read_single_time_step_outside_bounds_fails():

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()

    with pytest.raises(Exception):

        dfs.read(filename, item_numbers=[0, 3], time_steps=[100])


def test_get_number_of_time_steps():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()

    dfs.read(filename)
    assert dfs.get_number_of_time_steps() == 9


def test_get_number_of_time_steps_with_input_arg():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()

    dfs.read(filename, time_steps=[4])
    assert dfs.get_number_of_time_steps() == 9


def test_get_node_coords():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()
    dfs.read(filename)

    nc = dfs.get_node_coords()
    assert nc[0, 0] == 607031.4886285994


def test_get_element_coords():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()
    dfs.read(filename)

    ec = dfs.get_element_coords()
    assert ec[1, 1] == 6906790.5928664245


def test_find_closest_element_index():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()
    dfs.read(filename)

    idx = dfs.find_closest_element_index(606200, 6905480)
    assert idx == 317


def test_read_and_select_single_element():

    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()

    ds = dfs.read(filename)

    assert ds.data[0].shape == (9, 884)

    idx = dfs.find_closest_element_index(606200, 6905480)

    selds = ds.isel(idx=idx, axis=1)

    assert selds.data[0].shape == (9,)


def test_is_geo_UTM():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()
    dfs.read(filename)

    assert dfs.is_geo is False


def test_is_geo_LONGLAT():
    filename = os.path.join("tests", "testdata", "wind_north_sea.dfsu")
    dfs = Dfsu()
    dfs.read(filename)

    assert dfs.is_geo is True


def test_get_element_area_UTM():
    filename = os.path.join("tests", "testdata", "HD2D.dfsu")
    dfs = Dfsu()
    dfs.read(filename)

    areas = dfs.get_element_area()
    assert areas[0] == 4949.102548750438


def test_get_element_area_LONGLAT():
    filename = os.path.join("tests", "testdata", "wind_north_sea.dfsu")
    dfs = Dfsu()
    dfs.read(filename)

    areas = dfs.get_element_area()
    assert areas[0] == 139524218.81411952


def test_create(tmpdir):

    filename = os.path.join(tmpdir.dirname, "simple.dfsu")
    meshfilename = os.path.join("tests", "testdata", "odense_rough.mesh")

    msh = Mesh(meshfilename)
    # msh.read(meshfilename)
    n_elements = msh.number_of_elements
    d = np.zeros((1, n_elements))
    data = []
    data.append(d)

    items = [ItemInfo("Zeros")]

    dfs = Dfsu()

    dfs.create(meshfilename, filename, data, items=items)

    assert True


def test_create_invalid_data_closes_and_deletes_file(tmpdir):

    filename = os.path.join(tmpdir.dirname, "simple.dfsu")
    meshfilename = os.path.join("tests", "testdata", "odense_rough.mesh")

    msh = Mesh(meshfilename)

    n_elements = msh.number_of_elements
    d = np.zeros((1, n_elements - 1))

    assert d.shape[1] != n_elements
    data = []
    data.append(d)

    items = [ItemInfo("Bad data")]

    dfs = Dfsu()

    dfs.create(meshfilename, filename, data, items=items)

    assert not os.path.exists(filename)


def test_write_invalid_data_closes_file(tmpdir):

    infilename = os.path.join("tests", "testdata", "HD2D.dfsu")
    outfilename = os.path.join(tmpdir.dirname, "adjusted.dfsu")

    copyfile(infilename, outfilename)
    dfs = Dfsu()

    (data, t, items) = dfs.read(infilename)

    # Do some mistake, such as only trying to writing one item

    baddata = [data[0]]
    try:
        dfs.write(outfilename, baddata)
    except:
        print("Failed as expected")

    # Ok, it failed, try again, to write to the same file

    # Do arbitrary calculation
    newdata = data.copy()
    newdata[0] = newdata[0] * 2.0

    dfs.write(outfilename, newdata)

    (addata, t, items) = dfs.read(outfilename)

    assert addata[0][0, 0] == 2 * data[0][0, 0]
    assert addata[1][0, 0] == data[1][0, 0]
