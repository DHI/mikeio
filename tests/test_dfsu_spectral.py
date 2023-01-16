import pytest
import numpy as np
import matplotlib.pyplot as plt
import mikeio
from mikeio import eum
from mikecore.DfsuFile import DfsuFileType

from mikeio.dfsu.spectral import DfsuSpectral
from mikeio.spatial.FM_geometry import GeometryFMPointSpectrum, GeometryFMAreaSpectrum
import mikeio.spectral as spectral


@pytest.fixture
def dfsu_pt():
    filename = "tests/testdata/pt_spectra.dfsu"
    return mikeio.open(filename)


@pytest.fixture
def dfsu_line():
    filename = "tests/testdata/line_spectra.dfsu"
    return mikeio.open(filename)


@pytest.fixture
def dfsu_area():
    filename = "tests/testdata/area_spectra.dfsu"
    return mikeio.open(filename)


@pytest.fixture
def dfsu_area_sector():
    filename = "tests/testdata/MIKE21SW_dir_sector_area_spectra.dfsu"
    return mikeio.open(filename)


@pytest.fixture
def dfsu_pt_freq():
    filename = "tests/testdata/pt_freq_spectra.dfsu"
    return mikeio.open(filename)


@pytest.fixture
def dfsu_line_dir():
    filename = "tests/testdata/line_dir_spectra.dfsu"
    return mikeio.open(filename)


@pytest.fixture
def dfsu_area_freq():
    filename = "tests/testdata/area_freq_spectra.dfsu"
    return mikeio.open(filename)


def test_properties_pt_spectrum(dfsu_pt):
    dfs = dfsu_pt
    assert dfs.is_spectral
    assert dfs._type == DfsuFileType.DfsuSpectral0D
    assert dfs.n_frequencies == 25
    assert len(dfs.frequencies) == 25
    assert dfs.n_directions == 16
    assert len(dfs.directions) == 16


def test_properties_line_spectrum(dfsu_line):
    dfs = dfsu_line
    assert dfs.is_spectral
    assert dfs._type == DfsuFileType.DfsuSpectral1D
    assert dfs.n_frequencies == 25
    assert len(dfs.frequencies) == 25
    assert dfs.n_directions == 16
    assert len(dfs.directions) == 16
    assert dfs.n_nodes == 10
    assert dfs.n_elements == 9


def test_properties_area_spectrum(dfsu_area):
    dfs = dfsu_area
    assert dfs.is_spectral
    assert dfs._type == DfsuFileType.DfsuSpectral2D
    assert dfs.n_frequencies == 25
    assert len(dfs.frequencies) == 25
    assert dfs.n_directions == 16
    assert len(dfs.directions) == 16


def test_properties_line_dir_spectrum(dfsu_line_dir):
    dfs = dfsu_line_dir
    assert dfs.is_spectral
    assert dfs._type == DfsuFileType.DfsuSpectral1D
    assert dfs.n_frequencies == 0
    assert dfs.frequencies is None
    assert dfs.n_directions == 16
    assert len(dfs.directions) == 16


def test_properties_area_freq_spectrum(dfsu_area_freq):
    dfs = dfsu_area_freq
    assert dfs.is_spectral
    assert dfs._type == DfsuFileType.DfsuSpectral2D
    assert dfs.n_frequencies == 25
    assert len(dfs.frequencies) == 25
    assert dfs.n_directions == 0
    assert dfs.directions is None


def test_read_spectrum_pt(dfsu_pt):
    dfs = dfsu_pt
    ds1 = dfs.read(time=0)
    assert ds1.shape == (16, 25)
    assert ds1[0].type == eum.EUMType.Wave_energy_density
    assert ds1[0].to_numpy().max() == pytest.approx(0.03205060)

    ds2 = dfs.read()
    assert ds2.shape == (31, 16, 25)


def test_read_spectrum_area_sector(dfsu_area_sector):
    dfs = dfsu_area_sector
    assert dfs.n_frequencies == 25
    assert len(dfs.frequencies) == 25
    assert dfs.n_directions == 19
    assert len(dfs.directions) == 19
    assert dfs.directions.min() == pytest.approx(-90)
    assert dfs.directions.max() == pytest.approx(45)

    ds = dfs.read()
    assert ds.shape == (3, 40, 19, 25)
    assert ds[0].type == eum.EUMType.Wave_energy_density
    assert np.min(ds[0].to_numpy()) >= 0
    assert np.mean(ds[0].to_numpy()) == pytest.approx(0.001861494)


def test_read_pt_freq_spectrum(dfsu_pt_freq):
    dfs = dfsu_pt_freq
    assert dfs.n_directions == 0
    assert dfs.directions is None

    ds = dfs.read()
    assert ds.shape == (31, 25)
    assert ds[0].type == eum.EUMType.Directional_integrated_spectral_density
    assert np.min(ds[0].to_numpy()) >= 0
    assert np.mean(ds[0].to_numpy()) == pytest.approx(0.4229705970)


def test_read_area_freq_spectrum(dfsu_area_freq):
    dfs = dfsu_area_freq
    assert dfs.n_directions == 0
    assert dfs.directions is None

    ds = dfs.read()
    assert ds.shape == (3, 40, 25)
    assert ds.items[0].type == eum.EUMType.Directional_integrated_spectral_density
    assert np.min(ds[0].to_numpy()) >= 0
    assert np.mean(ds[0].to_numpy()) == pytest.approx(0.253988722)


def test_read_area_spectrum_elements(dfsu_area):
    dfs = dfsu_area
    ds1 = dfs.read()

    elems = [3, 4, 5, 6]
    ds2 = dfs.read(elements=elems)
    assert ds2.shape[1] == len(elems)
    assert np.all(ds1[0].to_numpy()[:, elems, ...] == ds2[0].to_numpy())


def test_read_area_spectrum_xy(dfsu_area):
    dfs = dfsu_area
    # ds1 = dfs.read()

    x, y = (2, 53)
    ds2 = dfs.read(x=x, y=y)
    assert isinstance(ds2.geometry, GeometryFMPointSpectrum)
    assert ds2.dims == ("time", "direction", "frequency")
    assert ds2.shape == (3, 16, 25)
    # TODO: add more asserts


def test_read_area_spectrum_area(dfsu_area):
    dfs = dfsu_area
    ds1 = dfs.read()
    assert ds1.n_frequencies == 25
    assert ds1.n_directions == 16

    bbox = (2.5, 51.8, 3.0, 52.2)
    ds2 = dfs.read(area=bbox)
    assert ds2.dims == ds1.dims
    assert ds2.shape == (3, 4, 16, 25)
    assert ds1.geometry._type == ds2.geometry._type
    assert ds2.n_frequencies == 25
    assert ds2.n_directions == 16
    assert ds2[0].n_frequencies == 25
    assert ds2[0].n_directions == 16


def test_read_spectrum_line_elements(dfsu_line):
    dfs = dfsu_line
    ds1 = dfs.read()

    nodes = [3, 4, 5, 6]
    ds2 = dfs.read(nodes=nodes)
    assert ds2.shape[1] == len(nodes)
    assert np.all(ds1[0].to_numpy()[:, nodes, ...] == ds2[0].to_numpy())


def test_spectrum_line_isel(dfsu_line):
    ds1 = dfsu_line.read()
    assert ds1.dims == ("time", "node", "direction", "frequency")

    nodes = [3, 4, 5]
    ds2 = dfsu_line.read(nodes=nodes)

    ds3 = ds1.isel(nodes, axis=1)
    assert ds3.shape == ds2.shape

    ds4 = ds1.isel(nodes, axis="node")
    assert ds4.shape == ds2.shape

    node = 3
    ds5 = dfsu_line.read(nodes=node)
    ds6 = ds1.isel(node, axis=1)
    assert ds6.shape == ds5.shape


def test_spectrum_line_getitem(dfsu_line):
    da1 = dfsu_line.read()[0]
    assert da1.dims == ("time", "node", "direction", "frequency")

    nodes = [3, 4, 5]
    da2 = dfsu_line.read(nodes=nodes)[0]
    da3 = da1[:, nodes]
    assert da3.shape == da2.shape

    node = 3
    da2 = dfsu_line.read(nodes=node)[0]
    da3 = da1[:, node]
    assert da3.shape == da2.shape


def test_spectrum_area_isel(dfsu_area):
    ds1 = dfsu_area.read()
    assert ds1.dims == ("time", "element", "direction", "frequency")

    elements = [7, 8, 9, 10, 11, 12]
    ds2 = dfsu_area.read(elements=elements)

    ds3 = ds1.isel(elements, axis=1)
    assert ds3.shape == ds2.shape

    ds4 = ds1.isel(elements, axis="element")
    assert ds4.shape == ds2.shape

    element = 3
    ds5 = dfsu_area.read(elements=element)
    ds6 = ds1.isel(element, axis=1)
    assert ds6.shape == ds5.shape


def test_spectrum_area_getitem(dfsu_area):
    da1 = dfsu_area.read()[0]
    assert da1.dims == ("time", "element", "direction", "frequency")

    elements = [7, 8, 9, 10, 11, 12]
    da2 = dfsu_area.read(elements=elements)[0]
    da3 = da1[:, elements]
    assert da3.shape == da2.shape

    element = 3
    da2 = dfsu_area.read(elements=element)[0]
    da3 = da1[:, element]
    assert da3.shape == da2.shape


def test_spectrum_area_sel_xy(dfsu_area):
    ds1 = dfsu_area.read()
    assert ds1.dims == ("time", "element", "direction", "frequency")

    element = 7
    xy = ds1.geometry.element_coordinates[element, :2]
    x, y = xy

    ds2 = dfsu_area.read(elements=element)  # reference
    assert ds2.geometry.x == x
    assert ds2.geometry.y == y
    assert isinstance(ds2.geometry, GeometryFMPointSpectrum)

    ds3 = ds1.sel(x=x, y=y)
    assert ds3.shape == ds2.shape
    assert ds3.geometry.x == x
    assert ds3.geometry.y == y
    assert np.all(ds3.to_numpy().ravel() == ds2.to_numpy().ravel())
    assert isinstance(ds3.geometry, GeometryFMPointSpectrum)


def test_spectrum_area_sel_area(dfsu_area):
    ds1 = dfsu_area.read()
    assert ds1.dims == ("time", "element", "direction", "frequency")

    bbox = [2.45, 52.2, 3, 52.4]
    ds2 = dfsu_area.read(area=bbox)  # reference
    assert isinstance(ds2.geometry, GeometryFMAreaSpectrum)
    assert ds2.geometry.n_elements == 6

    ds3 = ds1.sel(area=bbox)
    assert ds3.shape == ds2.shape
    assert ds3.geometry.n_elements == 6
    assert np.all(ds3.to_numpy().ravel() == ds2.to_numpy().ravel())
    assert isinstance(ds3.geometry, GeometryFMAreaSpectrum)


def test_read_spectrum_dir_line(dfsu_line_dir):
    dfs = dfsu_line_dir
    assert dfs.n_frequencies == 0
    assert dfs.frequencies is None

    ds1 = dfs.read(time=[0, 1])
    assert ds1.shape == (2, 10, 16)
    assert ds1.items[0].type == eum.EUMType.Frequency_integrated_spectral_density
    values = ds1[0].to_numpy()
    assert np.nanmin(values) >= 0
    assert np.nanmax(values) == pytest.approx(0.22447659)
    assert np.nanmean(values) == pytest.approx(0.02937540)
    assert np.all(np.isnan(ds1[0].to_numpy()[:, 0, :]))

    ds2 = dfs.read()
    assert ds2.shape == (4, 10, 16)


def test_calc_frequency_bin_sizes(dfsu_line):
    dfs = dfsu_line
    f = dfs.frequencies
    df = spectral._f_to_df(f)
    assert len(f) == len(df)
    assert df.max() < f.max()


def test_calc_Hm0_from_spectrum_line(dfsu_line):
    dfs: DfsuSpectral = dfsu_line
    assert dfs.n_elements == 9
    assert dfs.n_nodes == 10
    ds = dfs.read()
    assert ds.shape == (4, 10, 16, 25)

    Hm0 = dfs.calc_Hm0_from_spectrum(ds[0].to_numpy())
    assert Hm0.shape == (4, 10)
    assert np.all(~np.isnan(Hm0[:, 3:9]))
    assert np.all(np.isnan(Hm0[:, :3]))  # outside domain
    assert np.nanmin(Hm0) >= 0
    assert np.nanmax(Hm0) == pytest.approx(2.719780549)

    # DataArray works as well
    Hm0 = dfs.calc_Hm0_from_spectrum(ds[0])
    assert Hm0.shape == (4, 10)
    assert np.all(~np.isnan(Hm0[:, 3:9]))
    assert np.all(np.isnan(Hm0[:, :3]))  # outside domain
    assert np.nanmin(Hm0) >= 0
    assert np.nanmax(Hm0) == pytest.approx(2.719780549)


def test_calc_Hm0_from_spectrum_area(dfsu_area):
    dfs = dfsu_area
    ds = dfs.read()
    assert ds.shape == (3, 40, 16, 25)

    Hm0 = dfs.calc_Hm0_from_spectrum(ds[0].to_numpy())
    assert Hm0.shape == (3, 40)
    assert np.all(~np.isnan(Hm0))
    assert np.min(Hm0) >= 0
    assert np.max(Hm0) == pytest.approx(1.78410078776)


def test_plot_spectrum(dfsu_pt):
    dfs = dfsu_pt
    ds = dfs.read(time=0)
    spec = ds[0].to_numpy()
    dfs.plot_spectrum(spec, levels=3, add_colorbar=False)
    dfs.plot_spectrum(spec, vmin=0, cmap="Greys")
    dfs.plot_spectrum(spec, title="pt", plot_type="shaded")
    dfs.plot_spectrum(spec, r_as_periods=False, plot_type="contour")
    plt.close("all")


def test_plot_spectrum_sector(dfsu_area_sector):
    dfs = dfsu_area_sector
    ds = dfs.read(time=0)
    spec = ds[0].to_numpy()[0]
    dfs.plot_spectrum(spec)
    dfs.plot_spectrum(spec, rmax=10, vmin=0)
    dfs.plot_spectrum(spec, rmin=0, plot_type="patch")
    dfs.plot_spectrum(spec, r_as_periods=False, plot_type="contour")
    plt.close("all")


def test_plot_da_spectrum(dfsu_pt):
    dfs = dfsu_pt
    ds = dfs.read(time=0)
    da = ds[0]
    da.plot()
    #dfs.plot_spectrum(spec, levels=3, add_colorbar=False)
    #dfs.plot_spectrum(spec, vmin=0, cmap="Greys")
    #dfs.plot_spectrum(spec, title="pt", plot_type="shaded")
    #dfs.plot_spectrum(spec, r_as_periods=False, plot_type="contour")
    plt.close("all")

