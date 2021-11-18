import pytest
from mikeio import Dfsu
from mikecore.DfsuFile import DfsuFileType


# MIKE21SW_dir_sector_area_spectra.dfsu
# pt_freq_spectra.dfsu
# area_freq_spectra.dfsu
# line_dir_spectra.dfsu


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
    assert dfs.n_frequencies == 25
    assert len(dfs.frequencies) == 25
    assert dfs.n_directions == 16
    assert len(dfs.directions) == 16


def test_read_line_spectrum(dfsu_line):
    dfs = dfsu_line
    assert dfs.is_spectral
    assert dfs._type == DfsuFileType.DfsuSpectral1D
    assert dfs.n_frequencies == 25
    assert len(dfs.frequencies) == 25
    assert dfs.n_directions == 16
    assert len(dfs.directions) == 16
    assert dfs.n_nodes == 10
    assert dfs.n_elements == 9


def test_read_area_spectrum(dfsu_area):
    dfs = dfsu_area
    assert dfs.is_spectral
    assert dfs._type == DfsuFileType.DfsuSpectral2D
    assert dfs.n_frequencies == 25
    assert len(dfs.frequencies) == 25
    assert dfs.n_directions == 16
    assert len(dfs.directions) == 16


def test_read_spectrum_data():
    pass


def test_plot_spectrum():
    pass


def test_calc_frequency_bin_sizes():
    pass


def test_calc_Hm0_from_spectrum():
    pass


def test_calc_wave_parameters_from_spectrum():
    pass
