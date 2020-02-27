import os
import mikeio
from mikeio.generic import scale
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

    orgvalue = org.data[0][0]
    expected = 0.0
    scaledvalue = diffed.data[0][0]
    assert scaledvalue == pytest.approx(expected)
