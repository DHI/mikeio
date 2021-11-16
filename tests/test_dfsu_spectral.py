import pytest
from mikeio import Dfsu
from mikecore.DfsuFile import DfsuFileType


@pytest.fixture
def dfsu_pt():
    filename = "tests/testdata/pt_spectra.dfsu"
    return Dfsu(filename)


@pytest.fixture
def dfsu_line():
    filename = "tests/testdata/line_spectra.dfsu"
    return Dfsu(filename)


@pytest.fixture
def dfsu_area():
    filename = "tests/testdata/area_spectra.dfsu"
    return Dfsu(filename)


def test_read_pt_spectrum(dfsu_pt):
    dfs = dfsu_pt
    assert dfs.is_spectral
    assert dfs._type == DfsuFileType.DfsuSpectral0D


def test_read_line_spectrum(dfsu_line):
    dfs = dfsu_line
    assert dfs.is_spectral
    assert dfs._type == DfsuFileType.DfsuSpectral1D


def test_read_area_spectrum(dfsu_area):
    dfs = dfsu_area
    assert dfs.is_spectral
    assert dfs._type == DfsuFileType.DfsuSpectral2D
