import os
from shutil import copyfile
import numpy as np
import mikeio
from mikeio.generic import scale, diff, sum
import pytest


def test_add_constant(tmpdir):

    infilename = "tests/testdata/random.dfs0"
    outfilename = os.path.join(tmpdir.dirname, "add.dfs0")
    scale(infilename, outfilename, offset=100.0)

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue = org.data[0][0]
    expected = orgvalue + 100.0
    scaledvalue = scaled.data[0][0]
    assert scaledvalue == pytest.approx(expected)


def test_multiply_constant(tmpdir):

    infilename = "tests/testdata/random.dfs0"
    outfilename = os.path.join(tmpdir.dirname, "mult.dfs0")
    scale(infilename, outfilename, factor=1.5)

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue = org.data[0][0]
    expected = orgvalue * 1.5
    scaledvalue = scaled.data[0][0]
    assert scaledvalue == pytest.approx(expected)


def test_multiply_constant_single_item_number(tmpdir):

    infilename = "tests/testdata/wind_north_sea.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "mult.dfsu")
    scale(infilename, outfilename, factor=1.5, items=[0])

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue_speed = org.data[0][0][0]
    expected_speed = orgvalue_speed * 1.5
    scaledvalue_speed = scaled.data[0][0][0]
    assert scaledvalue_speed == pytest.approx(expected_speed)

    orgvalue_dir = org.data[1][0][0]
    expected_dir = orgvalue_dir
    scaledvalue_dir = scaled.data[1][0][0]
    assert scaledvalue_dir == pytest.approx(expected_dir)


def test_multiply_constant_single_item_name(tmpdir):

    infilename = "tests/testdata/wind_north_sea.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "multname.dfsu")
    scale(infilename, outfilename, factor=1.5, items=["Wind speed"])

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue_speed = org["Wind speed"][0, 0]
    expected_speed = orgvalue_speed * 1.5
    scaledvalue_speed = scaled["Wind speed"][0, 0]
    assert scaledvalue_speed == pytest.approx(expected_speed)

    orgvalue_dir = org["Wind direction"][0, 0]
    expected_dir = orgvalue_dir
    scaledvalue_dir = scaled["Wind direction"][0, 0]
    assert scaledvalue_dir == pytest.approx(expected_dir)


def test_diff_itself(tmpdir):

    infilename_1 = "tests/testdata/gebco_sound.dfs2"
    infilename_2 = "tests/testdata/gebco_sound.dfs2"
    outfilename = os.path.join(tmpdir.dirname, "diff.dfs2")
    
    diff(infilename_1, infilename_2, outfilename)

    org = mikeio.read(infilename_1)

    assert np.isnan(org["Elevation"][0][0,-1])

    diffed = mikeio.read(outfilename)

    diffedvalue = diffed["Elevation"][0, 0, 0]
    assert diffedvalue == pytest.approx(0.0)
    assert np.isnan(diffed["Elevation"][0][0,-1])

def test_sum_itself(tmpdir):

    infilename_1 = "tests/testdata/gebco_sound.dfs2"
    infilename_2 = "tests/testdata/gebco_sound.dfs2"
    outfilename = os.path.join(tmpdir.dirname, "diff.dfs2")
    
    sum(infilename_1, infilename_2, outfilename)

    org = mikeio.read(infilename_1)

    assert np.isnan(org["Elevation"][0][0,-1])

    summed = mikeio.read(outfilename)

    assert np.isnan(summed["Elevation"][0][0,-1])

def test_add_constant_delete_values_unchanged(tmpdir):

    infilename = "tests/testdata/gebco_sound.dfs2"
    outfilename = os.path.join(tmpdir.dirname, "adj.dfs2")
    scale(infilename, outfilename, offset=-2.1, items=["Elevation"])

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue = org["Elevation"][0, 0, 0]
    scaledvalue = scaled["Elevation"][0, 0, 0]
    assert scaledvalue == pytest.approx(orgvalue - 2.1)

    orgvalue = org["Elevation"][0, 100, 0]
    assert np.isnan(orgvalue)

    scaledvalue = scaled["Elevation"][0, 100, 0]
    assert np.isnan(scaledvalue)


def test_multiply_constant_delete_values_unchanged_2(tmpdir):

    infilename = "tests/testdata/random_two_item.dfs2"
    outfilename = os.path.join(tmpdir.dirname, "adj.dfs2")

    item_name = "testing water level"

    scale(infilename, outfilename, factor=1000.0, items=[item_name])

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue = org[item_name][0, 0, 0]
    scaledvalue = scaled[item_name][0, 0, 0]
    assert scaledvalue == pytest.approx(orgvalue * 1000.0)

    orgvalue = org[item_name][0, 10, 0]
    assert np.isnan(orgvalue)

    scaledvalue = scaled[item_name][0, 10, 0]
    assert np.isnan(scaledvalue)


def test_linear_transform(tmpdir):

    infilename = "tests/testdata/random.dfs0"
    outfilename = os.path.join(tmpdir.dirname, "linear.dfs0")
    scale(infilename, outfilename, offset=-20.0, factor=1.5)

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue = org.data[0][0]
    expected = orgvalue * 1.5 - 20.0
    scaledvalue = scaled.data[0][0]
    assert scaledvalue == pytest.approx(expected)


def test_linear_transform_dfsu(tmpdir):

    infilename = "tests/testdata/HD2D.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "linear.dfsu")
    scale(infilename, outfilename, offset=-20.0, factor=1.5)

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue = org.data[0][0]
    expected = orgvalue * 1.5 - 20.0
    scaledvalue = scaled.data[0][0]
    assert scaledvalue == pytest.approx(expected)


def test_sum_dfsu(tmpdir):

    infilename_a = "tests/testdata/HD2D.dfsu"
    infilename_b = "tests/testdata/HD2D.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "sum.dfsu")
    mikeio.generic.sum(infilename_a, infilename_b, outfilename)

    org = mikeio.read(infilename_a)

    summed = mikeio.read(outfilename)

    orgvalue = org.data[0][0]
    expected = orgvalue * 2
    scaledvalue = summed.data[0][0]
    assert scaledvalue == pytest.approx(expected)


def test_diff_dfsu(tmpdir):

    infilename_a = "tests/testdata/HD2D.dfsu"
    infilename_b = "tests/testdata/HD2D.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "diff.dfsu")
    mikeio.generic.diff(infilename_a, infilename_b, outfilename)

    org = mikeio.read(infilename_a)

    diffed = mikeio.read(outfilename)

    expected = 0.0
    scaledvalue = diffed.data[0][0]
    assert scaledvalue == pytest.approx(expected)


def test_concat_overlapping(tmpdir):
    infilename_a = "tests/testdata/tide1.dfs1"
    infilename_b = "tests/testdata/tide2.dfs1"
    outfilename = os.path.join(tmpdir.dirname, "concat.dfs1")

    mikeio.generic.concat([infilename_a, infilename_b], outfilename)

    ds = mikeio.read(outfilename)
    assert len(ds.time) == 145


def test_concat_files_gap_fail(tmpdir):
    infilename_a = "tests/testdata/tide1.dfs1"
    infilename_b = "tests/testdata/tide4.dfs1"
    outfilename = os.path.join(tmpdir.dirname, "concat.dfs1")
    with pytest.raises(Exception):
        mikeio.generic.concat([infilename_a, infilename_b], outfilename)


def test_concat_three_files(tmpdir):
    infiles = [
        "tests/testdata/tide1.dfs1",
        "tests/testdata/tide2.dfs1",
        "tests/testdata/tide4.dfs1",
    ]
    outfilename = os.path.join(tmpdir.dirname, "concat.dfs1")

    mikeio.generic.concat(infiles, outfilename)

    ds = mikeio.read(outfilename)
    assert len(ds.time) == (5 * 48 + 1)


def test_concat_closes_files(tmpdir):
    infiles = [
        "tests/testdata/tide1.dfs1",
        "tests/testdata/tide2.dfs1",
        "tests/testdata/tide4.dfs1",
    ]

    tmpfiles = []

    for file in infiles:
        tmp_path = os.path.join(tmpdir.dirname, os.path.basename(file))
        copyfile(file, tmp_path)
        tmpfiles.append(tmp_path)

    outfilename = os.path.join(tmpdir.dirname, "concat.dfs1")

    mikeio.generic.concat(tmpfiles, outfilename)

    for file in tmpfiles:
        os.remove(file)
        assert not os.path.exists(file)

    # ds = mikeio.read(outfilename)
    # assert len(ds.time) == (5 * 48 + 1)

