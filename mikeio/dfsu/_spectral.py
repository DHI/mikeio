from __future__ import annotations
from typing import Sequence, Sized, Any
from pathlib import Path

import numpy as np
import pandas as pd
from mikecore.DfsuFile import DfsuFile, DfsuFileType
from mikecore.DfsFileFactory import DfsFileFactory
from tqdm import trange

from ..dataset import DataArray, Dataset
from ..eum import ItemInfo, EUMUnit
from ..dfs._dfs import _get_item_info, _valid_item_numbers, _valid_timesteps
from .._spectral import calc_m0_from_spectrum
from ._dfsu import (
    _get_dfsu_info,
    get_elements_from_source,
    get_nodes_from_source,
    _validate_elements_and_geometry_sel,
)
from ..spatial import (
    GeometryFMAreaSpectrum,
    GeometryFMLineSpectrum,
    GeometryFMPointSpectrum,
)


class DfsuSpectral:
    """Dfsu for Spectral data.

    Parameters
    ----------
    filename:
        Path to dfsu file

    """

    show_progress = False

    def __init__(self, filename: str | Path) -> None:
        info = _get_dfsu_info(filename)
        self._filename = info.filename
        self._type = info.type
        self._deletevalue = info.deletevalue
        self._equidistant = info.equidistant
        self._start_time = info.start_time
        self._timestep = info.timestep
        self._n_timesteps = info.n_timesteps
        self._items = info.items
        self._geometry = self._read_geometry(self._filename)

    def __repr__(self) -> str:
        out = [f"<mikeio.{self.__class__.__name__}>"]

        if self._type is not DfsuFileType.DfsuSpectral0D:
            out.append(f"number of nodes: {self.geometry.n_nodes}")
            if self._type is not DfsuFileType.DfsuSpectral1D:
                out.append(f"number of elements: {self.geometry.n_elements}")
        if self.geometry.is_spectral:
            if self.geometry.n_directions > 0:
                out.append(f"number of directions: {self.geometry.n_directions}")
            if self.geometry.n_frequencies > 0:
                out.append(f"number of frequencies: {self.geometry.n_frequencies}")
        if self.geometry.projection_string:
            out.append(f"projection: {self.geometry.projection_string}")
        if self.n_items < 10:
            out.append("items:")
            for i, item in enumerate(self.items):
                out.append(f"  {i}:  {item}")
        else:
            out.append(f"number of items: {self.n_items}")
        if self.n_timesteps == 1:
            out.append(f"time: time-invariant file (1 step) at {self.time[0]}")
        else:
            out.append(
                f"time: {str(self.time[0])} - {str(self.time[-1])} ({self.n_timesteps} records)"
            )
        return str.join("\n", out)

    @property
    def geometry(
        self,
    ) -> GeometryFMPointSpectrum | GeometryFMLineSpectrum | GeometryFMAreaSpectrum:
        """Geometry."""
        return self._geometry

    @property
    def deletevalue(self) -> float:
        """File delete value."""
        return self._deletevalue

    @property
    def n_items(self) -> int:
        """Number of items."""
        return len(self.items)

    @property
    def items(self) -> list[ItemInfo]:
        """List of items."""
        return self._items

    @property
    def start_time(self) -> pd.Timestamp:
        """File start time."""
        return self._start_time

    @property
    def n_timesteps(self) -> int:
        """Number of time steps."""
        return self._n_timesteps

    @property
    def timestep(self) -> float:
        """Time step size in seconds."""
        return self._timestep

    @property
    def end_time(self) -> pd.Timestamp:
        """File end time."""
        if self._equidistant:
            return self.time[-1]
        else:
            # read the last timestep
            ds = self.read(items=0, time=-1)
            return ds.time[-1]

    @property
    def time(self) -> pd.DatetimeIndex:
        if self._equidistant:
            return pd.date_range(
                start=self.start_time,
                periods=self.n_timesteps,
                freq=f"{int(self.timestep)}s",
            )
        else:
            raise NotImplementedError(
                "Non-equidistant time axis. Read the data to get time."
            )

    @staticmethod
    def _read_geometry(
        filename: str,
    ) -> GeometryFMPointSpectrum | GeometryFMLineSpectrum | GeometryFMAreaSpectrum:
        dfs = DfsuFile.Open(filename)
        dfsu_type = DfsuFileType(dfs.DfsuFileType)

        directions = dfs.Directions
        if directions is not None:
            dir_unit = DfsuSpectral._get_direction_unit(filename)
            dir_conversion = 180.0 / np.pi if dir_unit == int(EUMUnit.radian) else 1.0
            directions = directions * dir_conversion

        frequencies = dfs.Frequencies

        # geometry
        if dfsu_type == DfsuFileType.DfsuSpectral0D:
            geometry: Any = GeometryFMPointSpectrum(
                frequencies=frequencies, directions=directions
            )  # No x,y coordinates
        else:
            # nc, codes, node_ids = get_nodes_from_source(dfs)
            node_table = get_nodes_from_source(dfs)
            el_table = get_elements_from_source(dfs)

            if dfsu_type == DfsuFileType.DfsuSpectral1D:
                geometry = GeometryFMLineSpectrum(
                    node_coordinates=node_table.coordinates,
                    element_table=el_table.connectivity,
                    codes=node_table.codes,
                    projection=dfs.Projection.WKTString,
                    dfsu_type=dfsu_type,
                    element_ids=el_table.ids,
                    node_ids=node_table.ids,
                    validate=False,
                    frequencies=frequencies,
                    directions=directions,
                )
            elif dfsu_type == DfsuFileType.DfsuSpectral2D:
                geometry = GeometryFMAreaSpectrum(
                    node_coordinates=node_table.coordinates,
                    element_table=el_table.connectivity,
                    codes=node_table.codes,
                    projection=dfs.Projection.WKTString,
                    dfsu_type=dfsu_type,
                    element_ids=el_table.ids,
                    node_ids=node_table.ids,
                    validate=False,
                    frequencies=frequencies,
                    directions=directions,
                )
        dfs.Close()
        return geometry

    @staticmethod
    def _get_direction_unit(filename: str) -> int:
        """Determine if the directional axis is in degrees or radians."""
        source = DfsFileFactory.DfsGenericOpen(filename)
        try:
            for static_item in iter(source.ReadStaticItemNext, None):
                if static_item.Name == "Direction":
                    return static_item.Quantity.Unit.value
        finally:
            source.Close()

        raise ValueError("Direction static item not found in the file.")

    @property
    def n_frequencies(self) -> int | None:
        """Number of frequencies."""
        return 0 if self.frequencies is None else len(self.frequencies)

    @property
    def frequencies(self) -> np.ndarray | None:
        """Frequency axis."""
        return self.geometry._frequencies

    @property
    def n_directions(self) -> int | None:
        """Number of directions."""
        return 0 if self.directions is None else len(self.directions)

    @property
    def directions(self) -> np.ndarray | None:
        """Directional axis."""
        return self.geometry._directions

    def _get_spectral_data_shape(
        self, n_steps: int, elements: Sized | None, dfsu_type: DfsuFileType
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
        dims = [] if n_steps == 1 else ["time"]
        n_freq = self.geometry.n_frequencies
        n_dir = self.geometry.n_directions
        shape: tuple[int, ...] = (n_dir, n_freq)
        if n_dir == 0:
            shape = (n_freq,)
        elif n_freq == 0:
            shape = (n_dir,)
        if dfsu_type == DfsuFileType.DfsuSpectral0D:
            read_shape = (n_steps, *shape)
        elif dfsu_type == DfsuFileType.DfsuSpectral1D:
            # node-based, FE-style
            n_nodes = self.geometry.n_nodes if elements is None else len(elements)
            if n_nodes == 1:
                read_shape = (n_steps, *shape)
            else:
                dims.append("node")
                read_shape = (n_steps, n_nodes, *shape)
            shape = (*shape, self.geometry.n_nodes)
        else:
            n_elems = self.geometry.n_elements if elements is None else len(elements)
            if n_elems == 1:
                read_shape = (n_steps, *shape)
            else:
                dims.append("element")
                read_shape = (n_steps, n_elems, *shape)
            shape = (*shape, self.geometry.n_elements)

        if n_dir > 1:
            dims.append("direction")
        if n_freq > 1:
            dims.append("frequency")

        return read_shape, shape, tuple(dims)

    def read(
        self,
        *,
        items: str | int | Sequence[str | int] | None = None,
        time: int | str | slice | Sequence[int] | None = None,
        elements: Sequence[int] | np.ndarray | int | None = None,
        nodes: Sequence[int] | np.ndarray | int | None = None,
        area: tuple[float, float, float, float] | None = None,
        x: float | None = None,
        y: float | None = None,
        keepdims: bool = False,
        dtype: Any = np.float32,
    ) -> Dataset:
        """Read data from a spectral dfsu file.

        Parameters
        ---------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time: int, str, datetime, pd.TimeStamp, sequence, slice or pd.DatetimeIndex, optional
            Read only selected time steps, by default None (=all)
        keepdims: bool, optional
            When reading a single time step only, should the time-dimension be kept
            in the returned Dataset? by default: False
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
        dtype: numpy.dtype, optional
            Data type to read. Default is np.float32

        Returns
        -------
        Dataset
            A Dataset with dimensions [t,elements/nodes,frequencies,directions]

        Examples
        --------
        >>> mikeio.read("tests/testdata/spectra/line_spectra.dfsu")
        <mikeio.Dataset>
        dims: (time:4, node:10, direction:16, frequency:25)
        time: 2017-10-27 00:00:00 - 2017-10-27 05:00:00 (4 records)
        geometry: DfsuSpectral1D (9 elements, 10 nodes)
        items:
          0:  Energy density <Wave energy density> (meter pow 2 sec per deg)

        >>> mikeio.read("tests/testdata/spectra/area_spectra.dfsu", time=-1)
        <mikeio.Dataset>
        dims: (element:40, direction:16, frequency:25)
        time: 2017-10-27 05:00:00 (time-invariant)
        geometry: DfsuSpectral2D (40 elements, 33 nodes)
        items:
          0:  Energy density <Wave energy density> (meter pow 2 sec per deg)

        """
        if dtype not in [np.float32, np.float64]:
            raise ValueError("Invalid data type. Choose np.float32 or np.float64")

        # Open the dfs file for reading
        # self._read_dfsu_header(self._filename)
        dfs = DfsuFile.Open(self._filename)

        single_time_selected, time_steps = _valid_timesteps(dfs, time)

        if self._type == DfsuFileType.DfsuSpectral2D:
            _validate_elements_and_geometry_sel(elements, area=area, x=x, y=y)
            if elements is None:
                elements = self._parse_geometry_sel(area=area, x=x, y=y)
        else:
            # TODO move to _parse_geometry_sel
            if (area is not None) or (x is not None) or (y is not None):
                raise ValueError(
                    f"Arguments area/x/y are not supported for {self._type}"
                )

        geometry, pts = self._parse_elements_nodes(elements, nodes)

        item_numbers = _valid_item_numbers(dfs.ItemInfo, items)
        items = _get_item_info(dfs.ItemInfo, item_numbers)
        n_items = len(item_numbers)

        deletevalue = self.deletevalue

        data_list = []

        n_steps = len(time_steps)
        read_shape, shape, dims = self._get_spectral_data_shape(
            n_steps, pts, self._type
        )
        for item in range(n_items):
            # Initialize an empty data block
            data: np.ndarray = np.ndarray(shape=read_shape, dtype=dtype)
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

                d = np.reshape(d, shape)
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
        return Dataset.from_numpy(
            data_list,
            time=time,
            items=items,
            geometry=geometry,
            dims=dims,
            validate=False,
        )

    def _parse_geometry_sel(
        self,
        area: tuple[float, float, float, float] | None,
        x: float | None,
        y: float | None,
    ) -> np.ndarray | None:
        """Parse geometry selection.

        Parameters
        ----------
        area : list[float], optional
            Read only data inside (horizontal) area given as a
            bounding box (tuple with left, lower, right, upper)
            or as list of coordinates for a polygon, by default None
        x : float, optional
            Read only data for elements containing the (x,y) points(s),
            by default None
        y : float, optional
            Read only data for elements containing the (x,y) points(s),
            by default None

        Returns
        -------
        list[int]
            List of element ids

        Raises
        ------
        ValueError
            If no elements are found in selection

        """
        elements = None

        if area is not None:
            assert isinstance(
                self.geometry, (GeometryFMLineSpectrum, GeometryFMAreaSpectrum)
            )
            elements = self.geometry._elements_in_area(area)

        if (x is not None) or (y is not None):
            assert isinstance(
                self.geometry, (GeometryFMLineSpectrum, GeometryFMAreaSpectrum)
            )
            elements = self.geometry.find_index(x=x, y=y)

        if (x is not None) or (y is not None) or (area is not None):
            # selection was attempted
            if (elements is None) or len(elements) == 0:
                raise ValueError("No elements in selection!")

        return elements

    def _parse_elements_nodes(
        self,
        elements: Sequence[int] | np.ndarray | int | None,
        nodes: Sequence[int] | np.ndarray | int | None,
    ) -> tuple[Any, Any]:
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
                geometry = self.geometry._nodes_to_geometry(nodes)  # type: ignore
                nodes = [nodes] if np.isscalar(nodes) else nodes  # type: ignore
            return geometry, nodes

        elif self._type == DfsuFileType.DfsuSpectral2D:
            if nodes is not None:
                raise ValueError(
                    "Reading specific nodes is only supported for DfsuSpectral1D"
                )
            if elements is None:
                geometry = self.geometry
            else:
                elements = (
                    [elements] if np.isscalar(elements) else list(elements)  # type: ignore
                )  # TODO check this
                geometry = self.geometry.elements_to_geometry(elements)  # type: ignore
            return geometry, elements  # type: ignore

        raise NotImplementedError(f"Not valid for type:{self._type}")

    def calc_Hm0_from_spectrum(
        self, spectrum: np.ndarray | DataArray, tail: bool = True
    ) -> np.ndarray:
        """Calculate significant wave height (Hm0) from spectrum.

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
