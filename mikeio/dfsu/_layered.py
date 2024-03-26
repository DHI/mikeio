from __future__ import annotations
from pathlib import Path
from typing import Any, Collection, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from mikecore.DfsuFile import DfsuFile, DfsuFileType
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import trange

from ..dataset import DataArray, Dataset
from ..dfs._dfs import (
    _get_item_info,
    _read_item_time_step,
    _valid_item_numbers,
    _valid_timesteps,
)
from ..eum import EUMType, ItemInfo
from .._interpolation import get_idw_interpolant, interp2d
from ..spatial import GeometryFM3D, GeometryFMVerticalProfile, GeometryPoint3D
from ..spatial._FM_utils import _plot_vertical_profile
from ._dfsu import (
    _get_dfsu_info,
    get_nodes_from_source,
    get_elements_from_source,
    _validate_elements_and_geometry_sel,
)

if TYPE_CHECKING:
    from ..spatial._FM_geometry_layered import Layer


class DfsuLayered:
    show_progress = False

    def __init__(self, filename: str | Path) -> None:
        info = _get_dfsu_info(filename)
        self._filename = info.filename
        self._type = info.type
        self._deletevalue = info.deletevalue
        self._time = info.time
        self._timestep = info.timestep
        self._geometry = self._read_geometry(self._filename)
        # 3d files have a zn item
        self._items = self._read_items(self._filename)

    def __repr__(self):
        out = [f"<mikeio.{self.__class__.__name__}>"]

        out.append(f"number of elements: {self.geometry.n_elements}")
        out.append(f"number of nodes: {self.geometry.n_nodes}")
        out.append(f"projection: {self.geometry.projection_string}")
        out.append(f"number of sigma layers: {self.geometry.n_sigma_layers}")
        if (
            self._type == DfsuFileType.DfsuVerticalProfileSigmaZ
            or self._type == DfsuFileType.Dfsu3DSigmaZ
        ):
            out.append(
                f"max number of z layers: {self.geometry.n_layers - self.geometry.n_sigma_layers}"
            )
        if self.n_items < 10:
            out.append("items:")
            for i, item in enumerate(self.items):
                out.append(f"  {i}:  {item}")
        else:
            out.append(f"number of items: {self.geometry.n_items}")
        if self.n_timesteps == 1:
            out.append(f"time: time-invariant file (1 step) at {self.time[0]}")
        else:
            out.append(
                f"time: {str(self.time[0])} - {str(self.time[-1])} ({self.n_timesteps} records)"
            )
        return str.join("\n", out)

    @property
    def deletevalue(self) -> float:
        """File delete value"""
        return self._deletevalue

    @property
    def n_items(self) -> int:
        """Number of items"""
        return len(self.items)

    @property
    def items(self) -> list[ItemInfo]:
        """List of items"""
        return self._items

    @property
    def start_time(self) -> pd.Timestamp:
        """File start time"""
        return self._time[0]

    @property
    def n_timesteps(self) -> int:
        """Number of time steps"""
        return len(self._time)

    @property
    def timestep(self) -> float:
        """Time step size in seconds"""
        return self._timestep

    @property
    def end_time(self) -> pd.Timestamp:
        """File end time"""
        return self._time[-1]

    @property
    def time(self) -> pd.DatetimeIndex:
        return self._time

    @property
    def geometry(self) -> GeometryFM3D | GeometryFMVerticalProfile:
        return self._geometry

    @staticmethod
    def _read_items(filename: str) -> list[ItemInfo]:
        dfs = DfsuFile.Open(filename)
        n_items = len(dfs.ItemInfo)
        first_idx = 1
        items = _get_item_info(
            dfs.ItemInfo,
            list(range(n_items - first_idx)),
            ignore_first=True,
        )
        dfs.Close()
        return items

    @staticmethod
    def _read_geometry(filename: str) -> GeometryFM3D | GeometryFMVerticalProfile:
        dfs = DfsuFile.Open(filename)
        dfsu_type = DfsuFileType(dfs.DfsuFileType)

        node_table = get_nodes_from_source(dfs)
        el_table = get_elements_from_source(dfs)

        geom_cls: Any = GeometryFM3D
        if dfsu_type in (
            DfsuFileType.DfsuVerticalProfileSigma,
            DfsuFileType.DfsuVerticalProfileSigmaZ,
        ):
            geom_cls = GeometryFMVerticalProfile

        geometry = geom_cls(
            node_coordinates=node_table.coordinates,
            element_table=el_table.connectivity,
            codes=node_table.codes,
            projection=dfs.Projection.WKTString,
            dfsu_type=dfsu_type,
            element_ids=el_table.ids,
            node_ids=node_table.ids,
            n_layers=dfs.NumberOfLayers,
            n_sigma=min(dfs.NumberOfSigmaLayers, dfs.NumberOfLayers),
            validate=False,
        )
        dfs.Close()
        return geometry

    @property
    def n_layers(self) -> int:
        """Maximum number of layers"""
        return self.geometry._n_layers

    @property
    def n_sigma_layers(self) -> int:
        """Number of sigma layers"""
        return self.geometry.n_sigma_layers

    @property
    def n_z_layers(self) -> int:
        """Maximum number of z-layers"""
        return self.n_layers - self.n_sigma_layers

    def read(
        self,
        *,
        items: str | int | Sequence[str | int] | None = None,
        time: int | str | slice | None = None,
        elements: Collection[int] | None = None,
        area: Tuple[float, float, float, float] | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        layers: int | Layer | Sequence[int] | None = None,
        keepdims: bool = False,
        dtype: Any = np.float32,
        error_bad_data: bool = True,
        fill_bad_data_value: float = np.nan,
    ) -> Dataset:
        """
        Read data from a dfsu file

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
            Read only data inside (horizontal) area given as a
            bounding box (tuple with left, lower, right, upper)
            or as list of coordinates for a polygon, by default None
        x: float, optional
            Read only data for elements containing the (x,y,z) points(s)
        y: float, optional
            Read only data for elements containing the (x,y,z) points(s)
        z: float, optional
            Read only data for elements containing the (x,y,z) points(s)
        layers: int, str, list[int], optional
            Read only data for specific layers, by default None
        elements: list[int], optional
            Read only selected element ids, by default None
        error_bad_data: bool, optional
            raise error if data is corrupt, by default True,
        fill_bad_data_value:
            fill value for to impute corrupt data, used in conjunction with error_bad_data=False
            default np.nan

        Returns
        -------
        Dataset
            A Dataset with data dimensions [t,elements]
        """
        if dtype not in [np.float32, np.float64]:
            raise ValueError("Invalid data type. Choose np.float32 or np.float64")

        dfs = DfsuFile.Open(self._filename)

        single_time_selected, time_steps = _valid_timesteps(dfs, time)

        _validate_elements_and_geometry_sel(
            elements, area=area, layers=layers, x=x, y=y, z=z
        )
        if elements is None:
            if (
                (x is not None)
                or (y is not None)
                or (area is not None)
                or (layers is not None)
            ):
                elements = self.geometry.find_index(  # type: ignore
                    x=x, y=y, z=z, area=area, layers=layers
                )
                if len(elements) == 0:
                    raise ValueError("No elements in selection!")

        geometry = (
            self.geometry
            if elements is None
            else self.geometry.elements_to_geometry(elements)
        )

        if isinstance(geometry, GeometryPoint3D):
            n_elems = 1
            n_nodes = 1
        else:
            n_elems = geometry.n_elements
            n_nodes = geometry.n_nodes
            node_ids = geometry.node_ids

        item_numbers = _valid_item_numbers(
            dfs.ItemInfo, items, ignore_first=self.geometry.is_layered
        )
        items = _get_item_info(
            dfs.ItemInfo, item_numbers, ignore_first=self.geometry.is_layered
        )
        item_numbers = [it + 1 for it in item_numbers]
        if layered_data := hasattr(geometry, "is_layered") and geometry.is_layered:
            item_numbers.insert(0, 0)
        n_items = len(item_numbers)

        deletevalue = self.deletevalue

        data_list = []

        n_steps = len(time_steps)
        item0_is_node_based = False
        for item in range(n_items):
            if layered_data and item == 0:
                item0_is_node_based = True
                data: np.ndarray = np.ndarray(shape=(n_steps, n_nodes), dtype=dtype)
            else:
                data = np.ndarray(shape=(n_steps, n_elems), dtype=dtype)
            data_list.append(data)

        if single_time_selected and not keepdims:
            data = data[0]

        time = self.time

        for i in trange(n_steps, disable=not self.show_progress):
            it = time_steps[i]
            for item in range(n_items):
                dfs, d = _read_item_time_step(
                    dfs=dfs,
                    filename=self._filename,
                    time=time,
                    item_numbers=item_numbers,
                    deletevalue=deletevalue,
                    shape=(data.shape[-1],),
                    item=item,
                    it=it,
                    error_bad_data=error_bad_data,
                    fill_bad_data_value=fill_bad_data_value,
                )

                if elements is not None:
                    if item == 0 and item0_is_node_based:
                        d = d[node_ids]
                    else:
                        d = d[elements]  # type: ignore

                if single_time_selected and not keepdims:
                    data_list[item] = d
                else:
                    data_list[item][i] = d

        time = self.time[time_steps]

        dfs.Close()

        dims = (
            ("time", "element")
            if not (single_time_selected and not keepdims)  # TODO extract variable
            else ("element",)
        )

        if elements is not None and len(elements) == 1:
            # squeeze point data
            dims = tuple([d for d in dims if d != "element"])
            data_list = [np.squeeze(d, axis=-1) for d in data_list]

        if layered_data:
            return Dataset(
                data_list[1:],  # skip zn item
                time,
                items,
                geometry=geometry,
                zn=data_list[0],
                dims=dims,
                validate=False,
            )
        else:
            return Dataset(
                data_list, time, items, geometry=geometry, dims=dims, validate=False
            )


class Dfsu2DV(DfsuLayered):
    def plot_vertical_profile(
        self, values, time_step=None, cmin=None, cmax=None, label="", **kwargs
    ):
        """
        Plot unstructured vertical profile

        Parameters
        ----------
        values: np.array
            value for each element to plot
        cmin: real, optional
            lower bound of values to be shown on plot, default:None
        cmax: real, optional
            upper bound of values to be shown on plot, default:None
        title: str, optional
            axes title
        label: str, optional
            colorbar label
        cmap: matplotlib.cm.cmap, optional
            colormap, default viridis
        figsize: (float, float), optional
            specify size of figure
        ax: matplotlib.axes, optional
            Adding to existing axis, instead of creating new fig

        Returns
        -------
        <matplotlib.axes>
        """
        if isinstance(values, DataArray):
            values = values.to_numpy()
        if time_step is not None:
            raise NotImplementedError(
                "Deprecated functionality. Instead, read as DataArray da, then use da.plot()"
            )

        g = self.geometry
        return _plot_vertical_profile(
            node_coordinates=g.node_coordinates,
            element_table=g.element_table,
            values=values,
            zn=None,
            is_geo=g.is_geo,
            cmin=cmin,
            cmax=cmax,
            label=label,
            **kwargs,
        )


class Dfsu3D(DfsuLayered):
    @property
    def geometry2d(self):
        """The 2d geometry for a 3d object"""
        return self.geometry.geometry2d

    def extract_surface_elevation_from_3d(self, n_nearest: int = 4) -> DataArray:
        """
        Extract surface elevation from a 3d dfsu file (based on zn)
        to a new 2d dfsu file with a surface elevation item.

        Parameters
        ---------
        n_nearest: int, optional
            number of points for spatial interpolation (inverse_distance), default=4
        """
        # validate input
        assert (
            self._type == DfsuFileType.Dfsu3DSigma
            or self._type == DfsuFileType.Dfsu3DSigmaZ
        )
        assert n_nearest > 0

        # make 2d nodes-to-elements interpolator
        top_el = self.geometry.top_elements
        geom = self.geometry.elements_to_geometry(top_el, node_layers="top")
        xye = geom.element_coordinates[:, 0:2]  # type: ignore
        xyn = geom.node_coordinates[:, 0:2]  # type: ignore
        tree2d = cKDTree(xyn)
        dist, node_ids = tree2d.query(xye, k=n_nearest)
        if n_nearest == 1:
            weights = None
        else:
            weights = get_idw_interpolant(dist)

        # read zn from 3d file and interpolate to element centers
        ds = self.read(items=0, keepdims=True)  # read only zn
        node_ids_surf, _ = self.geometry._get_nodes_and_table_for_elements(
            top_el, node_layers="top"
        )
        assert isinstance(ds[0]._zn, np.ndarray)
        zn_surf = ds[0]._zn[:, node_ids_surf]  # surface
        surf2d = interp2d(zn_surf, node_ids, weights)
        surf_da = DataArray(
            data=surf2d,
            time=ds.time,
            geometry=geom,
            item=ItemInfo(EUMType.Surface_Elevation),
        )

        return surf_da
