from typing import Union
import numpy as np
import pandas as pd
import warnings
from tqdm import trange

from mikecore.DfsuFile import DfsuFile, DfsuFileType

from .dfsu import _Dfsu
from .dataset import Dataset, DataArray
from .dfsutil import _get_item_info, _valid_item_numbers, _valid_timesteps
from .spectral_utils import plot_2dspectrum, calc_m0_from_spectrum


class DfsuSpectral(_Dfsu):
    @property
    def n_frequencies(self):
        """Number of frequencies"""
        return 0 if self.frequencies is None else len(self.frequencies)

    @property
    def frequencies(self):
        """Frequency axis"""
        return self._frequencies

    @property
    def n_directions(self):
        """Number of directions"""
        return 0 if self.directions is None else len(self.directions)

    @property
    def directions(self):
        """Directional axis"""
        return self._directions

    def _get_spectral_data_shape(self, n_steps: int, elements):
        dims = [] if n_steps == 1 else ["time"]
        n_freq = self.n_frequencies
        n_dir = self.n_directions
        shape = (n_dir, n_freq)
        if n_dir == 0:
            shape = [n_freq]
        elif n_freq == 0:
            shape = [n_dir]
        if self._type == DfsuFileType.DfsuSpectral0D:
            read_shape = (n_steps, *shape)
        elif self._type == DfsuFileType.DfsuSpectral1D:
            # node-based, FE-style
            n_nodes = self.n_nodes if elements is None else len(elements)
            if n_nodes == 1:
                read_shape = (n_steps, *shape)
            else:
                dims.append("node")
                read_shape = (n_steps, n_nodes, *shape)
            shape = (*shape, self.n_nodes)
        else:
            n_elems = self.n_elements if elements is None else len(elements)
            if n_elems == 1:
                read_shape = (n_steps, *shape)
            else:
                dims.append("element")
                read_shape = (n_steps, n_elems, *shape)
            shape = (*shape, self.n_elements)

        if n_dir > 1:
            dims.append("direction")
        if n_freq > 1:
            dims.append("frequency")

        return read_shape, shape, tuple(dims)

    def read(
        self,
        *,
        items=None,
        time=None,
        time_steps=None,
        elements=None,
        nodes=None,
        area=None,
        x=None,
        y=None,
        keepdims=False,
        dtype=np.float32,
    ) -> Dataset:
        """
        Read data from a spectral dfsu file

        Parameters
        ---------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time: str, int or list[int], optional
            Read only selected time_steps
        area: list[float], optional
            Read only data inside (horizontal) area (spectral area files
            only) given as a bounding box (tuple with left, lower, right, upper)
            or as list of coordinates for a polygon, by default None
        x, y: float, optional
            Read only data for elements containing the (x,y) points(s),
            by default None
        elements: list[int], optional
            Read only selected element ids (spectral area files only)
        nodes: list[int], optional
            Read only selected node ids (spectral line files only)

        Returns
        -------
        Dataset
            A Dataset with dimensions [t,elements/nodes,frequencies,directions]

        Examples
        --------
        >>> mikeio.read("line_spectra.dfsu")
        <mikeio.Dataset>
        Geometry: DfsuSpectral1D
        Dimensions: (time:4, node:10, frequency:16, direction:25)
        Time: 2017-10-27 00:00:00 - 2017-10-27 05:00:00 (4 records)
        Items:
        0:  Energy density <Wave energy density> (meter pow 2 sec per deg)

        >>> mikeio.read("area_spectra.dfsu", time=-1)
        <mikeio.Dataset>
        Geometry: DfsuSpectral2D
        Dimensions: (element:40, frequency:16, direction:25)
        Time: 2017-10-27 05:00:00 (time-invariant)
        Items:
        0:  Energy density <Wave energy density> (meter pow 2 sec per deg)
        """
        if dtype not in [np.float32, np.float64]:
            raise ValueError("Invalid data type. Choose np.float32 or np.float64")

        # Open the dfs file for reading
        # self._read_dfsu_header(self._filename)
        dfs = DfsuFile.Open(self._filename)

        self._n_timesteps = dfs.NumberOfTimeSteps
        if time_steps is not None:
            warnings.warn(
                FutureWarning(
                    "time_steps have been renamed to time, and will be removed in a future release"
                )
            )
            time = time_steps

        single_time_selected, time_steps = _valid_timesteps(dfs, time)

        if self._type == DfsuFileType.DfsuSpectral2D:
            self._validate_elements_and_geometry_sel(elements, area=area, x=x, y=y)
            if elements is None:
                elements = self._parse_geometry_sel(area=area, x=x, y=y)
        else:
            if (area is not None) or (x is not None) or (y is not None):
                raise ValueError(
                    f"Arguments area/x/y are not supported for {self._type}"
                )

        geometry, pts = self._parse_elements_nodes(elements, nodes)

        item_numbers = _valid_item_numbers(
            dfs.ItemInfo, items, ignore_first=self.is_layered
        )
        items = _get_item_info(dfs.ItemInfo, item_numbers, ignore_first=self.is_layered)
        n_items = len(item_numbers)

        deletevalue = self.deletevalue

        data_list = []

        n_steps = len(time_steps)
        read_shape, shape, dims = self._get_spectral_data_shape(n_steps, pts)
        for item in range(n_items):
            # Initialize an empty data block
            data = np.ndarray(shape=read_shape, dtype=dtype)
            data_list.append(data)

        t_seconds = np.zeros(n_steps, dtype=float)

        if single_time_selected and not keepdims:
            data = data[0]

        for i in trange(n_steps, disable=not self.show_progress):
            it = time_steps[i]
            for item in range(n_items):

                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)
                d = itemdata.Data
                d[d == deletevalue] = np.nan

                d = np.reshape(d, newshape=shape)
                if self._type != DfsuFileType.DfsuSpectral0D:
                    d = np.moveaxis(d, -1, 0)

                if pts is not None:
                    d = d[pts, ...]

                if single_time_selected and not keepdims:
                    data_list[item] = d
                else:
                    data_list[item][i] = d

            t_seconds[i] = itemdata.Time

        dfs.Close()

        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)
        return Dataset(
            data_list, time, items, geometry=geometry, dims=dims, validate=False
        )

    def _parse_elements_nodes(self, elements, nodes):
        if self._type == DfsuFileType.DfsuSpectral0D:
            if elements is not None or nodes is not None:
                raise ValueError(
                    "Reading specific elements/nodes is not supported for DfsuSpectral0D"
                )
            geometry = self.geometry
            return geometry, None

        elif self._type == DfsuFileType.DfsuSpectral1D:
            if elements is not None:
                raise ValueError(
                    "Reading specific elements is not supported for DfsuSpectral1D"
                )
            if nodes is None:
                geometry = self.geometry
            else:
                geometry = self.geometry._nodes_to_geometry(nodes)
                nodes = [nodes] if np.isscalar(nodes) else nodes
            return geometry, nodes

        elif self._type == DfsuFileType.DfsuSpectral2D:
            if nodes is not None:
                raise ValueError(
                    "Reading specific nodes is only supported for DfsuSpectral1D"
                )
            if elements is None:
                geometry = self.geometry
            else:
                elements = [elements] if np.isscalar(elements) else elements
                geometry = self.geometry.elements_to_geometry(elements)
            return geometry, elements

    def plot_spectrum(
        self,
        spectrum,
        plot_type="contourf",
        title=None,
        label=None,
        cmap="Reds",
        vmin=1e-5,
        vmax=None,
        r_as_periods=True,
        rmin=None,
        rmax=None,
        levels=None,
        figsize=(7, 7),
        add_colorbar=True,
    ):
        """
        Plot spectrum in polar coordinates

        Parameters
        ----------
        spectrum: np.array
            spectral values as 2d array with dimensions: directions, frequencies
        plot_type: str, optional
            type of plot: 'contour', 'contourf', 'patch', 'shaded',
            by default: 'contourf'
        title: str, optional
            axes title
        label: str, optional
            colorbar label (or title if contour plot)
        cmap: matplotlib.cm.cmap, optional
            colormap, default Reds
        vmin: real, optional
            lower bound of values to be shown on plot, default: 1e-5
        vmax: real, optional
            upper bound of values to be shown on plot, default:None
        r_as_periods: bool, optional
            show radial axis as periods instead of frequency, default: True
        rmin: float, optional
            mininum frequency/period to be shown, default: None
        rmax: float, optional
            maximum frequency/period to be shown, default: None
        levels: int, list(float), optional
            for contour plots: how many levels, default:10
            or a list of discrete levels e.g. [0.03, 0.04, 0.05]
        figsize: (float, float), optional
            specify size of figure, default (7, 7)
        add_colorbar: bool, optional
            Add colorbar to plot, default True

        Returns
        -------
        <matplotlib.axes>

        Examples
        --------
        >>> dfs = Dfsu("area_spectrum.dfsu")
        >>> ds = dfs.read(items="Energy density")
        >>> spectrum = ds[0][0, 0, :, :] # first timestep, element 0
        >>> dfs.plot_spectrum(spectrum, plot_type="patch")

        >>> dfs.plot_spectrum(spectrum, rmax=9, title="Wave spectrum T<9s")
        """
        if isinstance(spectrum, DataArray):
            spectrum = spectrum.to_numpy()

        return plot_2dspectrum(
            spectrum,
            frequencies=self.frequencies,
            directions=self.directions,
            plot_type=plot_type,
            title=title,
            label=label,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            r_as_periods=r_as_periods,
            rmin=rmin,
            rmax=rmax,
            levels=levels,
            figsize=figsize,
            add_colorbar=add_colorbar,
        )

    def calc_Hm0_from_spectrum(
        self, spectrum: Union[np.ndarray, DataArray], tail=True
    ) -> np.ndarray:
        """Calculate significant wave height (Hm0) from spectrum

        Parameters
        ----------
        spectrum : np.ndarray, DataArray
            frequency or direction-frequency spectrum
        tail : bool, optional
            Should a parametric spectral tail be added in the computations? by default True

        Returns
        -------
        np.ndarray
            significant wave height values
        """
        if isinstance(spectrum, DataArray):
            m0 = calc_m0_from_spectrum(
                spectrum.to_numpy(),
                self.frequencies,
                self.directions,
                tail,
            )
        else:

            m0 = calc_m0_from_spectrum(
                spectrum, self.frequencies, self.directions, tail
            )
        return 4 * np.sqrt(m0)
