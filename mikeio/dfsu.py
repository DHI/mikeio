import os
import numpy as np
import pandas as pd
import warnings

from datetime import datetime, timedelta
from functools import wraps
from tqdm import trange

from mikecore.eum import eumUnit, eumQuantity
from mikecore.DfsFactory import DfsFactory
from mikecore.DfsuBuilder import DfsuBuilder
from mikecore.DfsuFile import DfsuFile, DfsuFileType
from mikecore.MeshFile import MeshFile
from mikecore.MeshBuilder import MeshBuilder

from mikeio.spatial.utils import xy_to_bbox

from .base import EquidistantTimeSeries
from .dfsutil import _get_item_info, _valid_item_numbers, _valid_timesteps
from .dataset import Dataset, DataArray
from .dfs0 import Dfs0
from .eum import ItemInfo, EUMType, EUMUnit
from .spatial.FM_geometry import (
    GeometryFM,
    GeometryFM3D,
    GeometryFMVerticalProfile,
    GeometryFMPointSpectrum,
    GeometryFMPointSpectrum,
    GeometryFMAreaSpectrum,
    GeometryFMLineSpectrum,
)
from .spatial.FM_utils import _plot_map
from .spatial.grid_geometry import Grid2D


def _write_dfsu(filename: str, data: Dataset):

    filename = str(filename)

    if len(data.time) == 1:
        dt = 1  # TODO is there any sensible default?
    else:
        dt = (data.time[1] - data.time[0]).total_seconds()
    n_time_steps = len(data.time)

    geometry = data.geometry
    dfsu_filetype = DfsuFileType.Dfsu2D

    if geometry.is_layered:
        dfsu_filetype = geometry._type.value

    xn = geometry.node_coordinates[:, 0]
    yn = geometry.node_coordinates[:, 1]
    zn = geometry.node_coordinates[:, 2]

    elem_table = [np.array(e) + 1 for e in geometry.element_table]

    builder = DfsuBuilder.Create(dfsu_filetype)
    if dfsu_filetype != DfsuFileType.Dfsu2D:
        builder.SetNumberOfSigmaLayers(geometry.n_sigma_layers)

    builder.SetNodes(xn, yn, zn, geometry.codes)
    builder.SetElements(elem_table)

    factory = DfsFactory()
    proj = factory.CreateProjection(geometry.projection_string)
    builder.SetProjection(proj)
    builder.SetTimeInfo(data.time[0], dt)
    builder.SetZUnit(eumUnit.eumUmeter)

    if dfsu_filetype != DfsuFileType.Dfsu2D:
        builder.SetNumberOfSigmaLayers(geometry.n_sigma_layers)

    for item in data.items:
        builder.AddDynamicItem(item.name, eumQuantity.Create(item.type, item.unit))

    dfs = builder.CreateFile(filename)

    for i in range(n_time_steps):
        if geometry.is_layered:
            if "time" in data.dims:
                zn = data._zn[i]
            else:
                zn = data._zn
            dfs.WriteItemTimeStepNext(0, zn.astype(np.float32))
        for da in data:
            if "time" in data.dims:
                d = da.to_numpy()[i, :]
            else:
                d = da.to_numpy()
            d[np.isnan(d)] = data.deletevalue
            dfs.WriteItemTimeStepNext(0, d.astype(np.float32))
    dfs.Close()


class _UnstructuredFile:
    """
    _UnstructuredFile is base class for Mesh and Dfsu
    has file handle, items and timesteps and reads file header
    """

    show_progress = False

    def __init__(self) -> None:
        self._type = None  # -1: mesh, 0: 2d-dfsu, 4:dfsu3dsigma, ...
        self._geometry = None
        self._geom2d = None
        # self._shapely_domain_obj = None

        self._filename = None
        self._source = None
        self._deletevalue = None

        self._n_timesteps = None
        self._start_time = None
        self._timestep_in_seconds = None

        self._items = None
        self._dtype = np.float64

    def __repr__(self):
        out = []
        type_name = "Flexible Mesh" if self._type is None else self.type_name
        out.append(type_name)

        if self._type is not DfsuFileType.DfsuSpectral0D:
            if self._type is not DfsuFileType.DfsuSpectral1D:
                out.append(f"number of elements: {self.n_elements}")
            out.append(f"number of nodes: {self.n_nodes}")
        if self.is_spectral:
            if self.n_directions > 0:
                out.append(f"number of directions: {self.n_directions}")
            if self.n_frequencies > 0:
                out.append(f"number of frequencies: {self.n_frequencies}")
        if self.geometry.projection_string:
            out.append(f"projection: {self.projection_string}")
        if self.is_layered:
            out.append(f"number of sigma layers: {self.n_sigma_layers}")
        if (
            self._type == DfsuFileType.DfsuVerticalProfileSigmaZ
            or self._type == DfsuFileType.Dfsu3DSigmaZ
        ):
            out.append(f"max number of z layers: {self.n_layers - self.n_sigma_layers}")
        if hasattr(self, "items") and self.items is not None:
            if self.n_items < 10:
                out.append("items:")
                for i, item in enumerate(self.items):
                    out.append(f"  {i}:  {item}")
            else:
                out.append(f"number of items: {self.n_items}")
        if self._n_timesteps is not None:
            if self._n_timesteps == 1:
                out.append(f"time: time-invariant file (1 step) at {self._start_time}")
            else:
                out.append(
                    f"time: {self._n_timesteps} steps with dt={self._timestep_in_seconds}s"
                )
                out.append(f"      {self._start_time} -- {self.end_time}")
        return str.join("\n", out)

    def _read_header(self, input):
        if isinstance(input, DfsuFile):
            # input is a dfsu file object (already open)
            self._read_dfsu_header(input)
            return

        filename = input
        if not os.path.isfile(filename):
            raise Exception(f"file {filename} does not exist!")

        _, ext = os.path.splitext(filename)

        if ext == ".mesh":
            self._read_mesh_header(filename)

        elif ext == ".dfsu":
            dfs = DfsuFile.Open(filename)
            self._read_dfsu_header(dfs)
        else:
            raise Exception(f"Filetype {ext} not supported (mesh,dfsu)")

    def _read_mesh_header(self, filename):
        """
        Read header of mesh file and set object properties
        """
        msh = MeshFile.ReadMesh(filename)
        self._source = msh
        self._type = None  # =DfsuFileType.Mesh

        nc, codes, node_ids = self._get_nodes_from_source(msh)
        el_table, el_ids = self._get_elements_from_source(msh)

        self._geometry = GeometryFM(
            node_coordinates=nc,
            element_table=el_table,
            codes=codes,
            projection=msh.ProjectionString,
            dfsu_type=self._type,
            element_ids=el_ids,
            node_ids=node_ids,
            validate=False,
        )

    def _read_dfsu_header(self, dfs):
        """
        Read header of dfsu file and set object properties
        """
        self._source = dfs
        self._type = DfsuFileType(dfs.DfsuFileType)
        self._deletevalue = dfs.DeleteValueFloat

        if self.is_spectral:
            dir = dfs.Directions
            self._directions = None if dir is None else dir * (180 / np.pi)
            self._frequencies = dfs.Frequencies

        # geometry
        if self._type == DfsuFileType.DfsuSpectral0D:
            self._geometry = GeometryFMPointSpectrum(
                frequencies=self.frequencies, directions=self.directions
            )
        else:
            nc, codes, node_ids = self._get_nodes_from_source(dfs)
            el_table, el_ids = self._get_elements_from_source(dfs)

            if self.is_layered:
                geom_cls = GeometryFM3D
                if self._type in (
                    DfsuFileType.DfsuVerticalProfileSigma,
                    DfsuFileType.DfsuVerticalProfileSigmaZ,
                ):
                    geom_cls = GeometryFMVerticalProfile

                self._geometry = geom_cls(
                    node_coordinates=nc,
                    element_table=el_table,
                    codes=codes,
                    projection=dfs.Projection.WKTString,
                    dfsu_type=self._type,
                    element_ids=el_ids,
                    node_ids=node_ids,
                    n_layers=dfs.NumberOfLayers,
                    n_sigma=dfs.NumberOfSigmaLayers,
                    validate=False,
                )
            elif self._type == DfsuFileType.DfsuSpectral1D:
                self._geometry = GeometryFMLineSpectrum(
                    node_coordinates=nc,
                    element_table=el_table,
                    codes=codes,
                    projection=dfs.Projection.WKTString,
                    dfsu_type=self._type,
                    element_ids=el_ids,
                    node_ids=node_ids,
                    validate=False,
                    frequencies=self.frequencies,
                    directions=self.directions,
                )
            elif self._type == DfsuFileType.DfsuSpectral2D:
                self._geometry = GeometryFMAreaSpectrum(
                    node_coordinates=nc,
                    element_table=el_table,
                    codes=codes,
                    projection=dfs.Projection.WKTString,
                    dfsu_type=self._type,
                    element_ids=el_ids,
                    node_ids=node_ids,
                    validate=False,
                    frequencies=self.frequencies,
                    directions=self.directions,
                )
            else:
                self._geometry = GeometryFM(
                    node_coordinates=nc,
                    element_table=el_table,
                    codes=codes,
                    projection=dfs.Projection.WKTString,
                    dfsu_type=self._type,
                    element_ids=el_ids,
                    node_ids=node_ids,
                    validate=False,
                )

        # items
        n_items = len(dfs.ItemInfo)
        first_idx = 1 if self.is_layered else 0
        self._items = _get_item_info(
            dfs.ItemInfo, list(range(n_items - first_idx)), ignore_first=self.is_layered
        )

        # time
        self._start_time = dfs.StartDateTime
        self._n_timesteps = dfs.NumberOfTimeSteps
        self._timestep_in_seconds = dfs.TimeStepInSeconds

        dfs.Close()

    @staticmethod
    def _get_nodes_from_source(source):
        xn = source.X
        yn = source.Y
        zn = source.Z
        nc = np.column_stack([xn, yn, zn])
        codes = np.array(list(source.Code))
        node_ids = source.NodeIds - 1
        return nc, codes, node_ids

    @staticmethod
    def _get_elements_from_source(source):
        element_table = _UnstructuredFile._get_element_table_from_mikecore(
            source.ElementTable
        )
        element_ids = source.ElementIds - 1
        return element_table, element_ids

    @staticmethod
    def _offset_element_table_by(element_table, offset, copy=True):
        offset = int(offset)
        new_elem_table = element_table.copy() if copy else element_table
        for j in range(len(element_table)):
            new_elem_table[j] = element_table[j] + offset
        return new_elem_table

    @staticmethod
    def _get_element_table_from_mikecore(element_table):
        return _UnstructuredFile._offset_element_table_by(element_table, -1)

    @staticmethod
    def _element_table_to_mikecore(element_table):
        return _UnstructuredFile._offset_element_table_by(element_table, 1)

    @property
    def type_name(self):
        """Type name, e.g. Mesh, Dfsu2D"""
        return self._type.name if self._type else "Mesh"

    @property
    def geometry(self):
        return self._geometry

    @property
    def _geometry2d(self):
        """The 2d geometry for a 3d object or geometry for a 2d object"""
        if self.geometry.is_2d:
            return self.geometry
        if self._geom2d is None:
            self._geom2d = self.geometry.to_2d_geometry()
        return self._geom2d

    @property
    def n_nodes(self):
        """Number of nodes"""
        return self.geometry.n_nodes

    @property
    def node_coordinates(self):
        """Coordinates (x,y,z) of all nodes"""
        return self.geometry.node_coordinates

    @property
    def node_ids(self):
        return self.geometry.node_ids

    @property
    def n_elements(self):
        """Number of elements"""
        return self.geometry.n_elements

    @property
    def element_ids(self):
        return self.geometry.element_ids

    @property
    def codes(self):
        warnings.warn(
            "property codes is deprecated, use .geometry.codes instead",
            FutureWarning,
        )
        return self.geometry.codes

    @codes.setter
    def codes(self, v):
        if len(v) != self.n_nodes:
            raise ValueError(f"codes must have length of nodes ({self.n_nodes})")
        self._geometry._codes = np.array(v, dtype=np.int32)

    @property
    def valid_codes(self):
        """Unique list of node codes"""
        return list(set(self.codes))

    @property
    def boundary_codes(self):
        """Unique list of boundary codes"""
        return [code for code in self.valid_codes if code > 0]

    @property
    def projection_string(self):
        """The projection string"""
        return self.geometry.projection_string

    @property
    def is_geo(self):
        """Are coordinates geographical (LONG/LAT)?"""
        return self.geometry.projection_string == "LONG/LAT"

    @property
    def is_local_coordinates(self):
        """Are coordinates relative (NON-UTM)?"""
        return self.geometry.projection_string == "NON-UTM"

    @property
    def element_table(self):
        """Element to node connectivity"""
        return self.geometry.element_table

    @property
    def max_nodes_per_element(self):
        """The maximum number of nodes for an element"""
        return self.geometry.max_nodes_per_element

    @property
    def is_2d(self):
        """Type is either mesh or Dfsu2D (2 horizontal dimensions)"""
        return self._type in (
            DfsuFileType.Dfsu2D,
            DfsuFileType.DfsuSpectral2D,
            None,
        )

    @property
    def is_layered(self):
        """Type is layered dfsu (3d, vertical profile or vertical column)"""
        return self._type in (
            DfsuFileType.DfsuVerticalColumn,
            DfsuFileType.DfsuVerticalProfileSigma,
            DfsuFileType.DfsuVerticalProfileSigmaZ,
            DfsuFileType.Dfsu3DSigma,
            DfsuFileType.Dfsu3DSigmaZ,
        )

    @property
    def is_spectral(self):
        """Type is spectral dfsu (point, line or area spectrum)"""
        return self._type in (
            DfsuFileType.DfsuSpectral0D,
            DfsuFileType.DfsuSpectral1D,
            DfsuFileType.DfsuSpectral2D,
        )

    @property
    def is_tri_only(self):
        """Does the mesh consist of triangles only?"""
        return self.geometry.is_tri_only

    @property
    def boundary_polylines(self):
        """Lists of closed polylines defining domain outline"""
        return self.geometry.boundary_polylines

    def get_node_coords(self, code=None):
        """Get the coordinates of each node.

        Parameters
        ----------
        code: int
            Get only nodes with specific code, e.g. land == 1

        Returns
        -------
        np.array
            x,y,z of each node
        """
        nc = self.node_coordinates
        if code is not None:
            if code not in self.valid_codes:
                print(
                    f"Selected code: {code} is not valid. Valid codes: {self.valid_codes}"
                )
                raise Exception
            return nc[self.codes == code]
        return nc

    @wraps(GeometryFM.elements_to_geometry)
    def elements_to_geometry(self, elements, node_layers="all"):
        return self.geometry.elements_to_geometry(elements, node_layers)

    @property
    def element_coordinates(self):
        """Center coordinates of each element"""
        return self.geometry.element_coordinates

    @wraps(GeometryFM3D.calc_element_coordinates)
    def calc_element_coordinates(self, elements=None, zn=None):
        return self.geometry.calc_element_coordinates(elements, zn)

    @wraps(GeometryFM.contains)
    def contains(self, points):
        return self.geometry.contains(points)

    def get_overset_grid(self, dx=None, dy=None, nx=None, ny=None, buffer=None):
        """get a 2d grid that covers the domain by specifying spacing or shape

        Parameters
        ----------
        dx : float, optional
            grid resolution in x-direction (or in x- and y-direction)
        dy : float, optional
            grid resolution in y-direction
        nx : int, optional
            number of points in x-direction,
            by default None (the value will be inferred)
        ny : int, optional
            number of points in y-direction,
            by default None (the value will be inferred)
        buffer : float, optional
            positive to make the area larger, default=0
            can be set to a small negative value to avoid NaN
            values all around the domain.

        Returns
        -------
        <mikeio.Grid2D>
            2d grid
        """
        nc = self._geometry2d.node_coordinates
        bbox = xy_to_bbox(nc, buffer=buffer)
        return Grid2D(
            bbox=bbox,
            dx=dx,
            dy=dy,
            nx=nx,
            ny=ny,
            projection=self.geometry.projection_string,
        )

    @wraps(GeometryFM.get_2d_interpolant)
    def get_2d_interpolant(
        self, xy, n_nearest: int = 1, extrapolate=False, p=2, radius=None
    ):
        warnings.warn(
            FutureWarning(
                "get_2d_interpolant have been deprecated, please use DataArray.interp() or DataArray.interp_like() instead"
            )
        )
        return self.geometry.get_2d_interpolant(xy, n_nearest, extrapolate, p, radius)

    @wraps(GeometryFM.interp2d)
    def interp2d(self, data, elem_ids, weights=None, shape=None):

        warnings.warn(
            FutureWarning(
                "interp2d have been deprecated, please use DataArray.interp() or DataArray.interp_like() instead"
            )
        )
        return self.geometry.interp2d(data, elem_ids, weights, shape)

    @wraps(GeometryFM.find_nearest_elements)
    def find_nearest_elements(
        self, x, y=None, z=None, layer=None, n_nearest=1, return_distances=False
    ):
        warnings.warn(
            FutureWarning(
                "find_nearest_elements have been deprecated, please use .geometry.find_index() instead"
            )
        )
        return self.geometry.find_nearest_elements(
            x, y, z, layer, n_nearest, return_distances
        )

    @wraps(GeometryFM.get_element_area)
    def get_element_area(self):
        return self.geometry.get_element_area()

    @wraps(GeometryFM.to_shapely)
    def to_shapely(self):
        return self.geometry.to_shapely()

    @wraps(GeometryFM.get_node_centered_data)
    def get_node_centered_data(self, data, extrapolate=True):
        return self.geometry.get_node_centered_data(data, extrapolate)

    def plot(
        self,
        z=None,
        elements=None,
        plot_type="patch",
        title=None,
        label=None,
        cmap=None,
        vmin=None,
        vmax=None,
        levels=None,
        n_refinements=0,
        show_mesh=True,
        show_outline=True,
        figsize=None,
        ax=None,
        add_colorbar=True,
    ):

        warnings.warn(
            FutureWarning(
                "Dfsu.plot() have been deprecated, please use DataArray.plot() instead"
            )
        )
        if elements is None:
            if self.is_2d:
                geometry = self.geometry
            else:
                geometry = self._geometry2d
        else:
            # spatial subset
            if self.is_2d:
                geometry = self.geometry.elements_to_geometry(elements)
            else:
                geometry = self.geometry.elements_to_geometry(
                    elements, node_layers="bottom"
                )
        if z is not None:
            if isinstance(z, DataArray):
                z = z.to_numpy().copy()
            if isinstance(z, Dataset) and len(z) == 1:  # if single-item Dataset
                z = z[0].to_numpy().copy()

        return _plot_map(
            node_coordinates=geometry.node_coordinates,
            element_table=geometry.element_table,
            element_coordinates=geometry.element_coordinates,
            boundary_polylines=self.boundary_polylines,
            projection=geometry.projection,
            z=z,
            plot_type=plot_type,
            title=title,
            label=label,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            n_refinements=n_refinements,
            show_mesh=show_mesh,
            show_outline=show_outline,
            figsize=figsize,
            ax=ax,
            add_colorbar=add_colorbar,
        )


class _Dfsu(_UnstructuredFile, EquidistantTimeSeries):
    def __init__(self, filename, dfs=None, dtype=np.float32):
        """
        Create a Dfsu object

        Parameters
        ---------
        filename: str
            dfsu or mesh filename
        dfs :
        dtype: np.dtype, optional
            default np.float32, valid options are np.float32, np.float64
        """
        if dtype not in [np.float32, np.float64]:
            raise ValueError("Invalid data type. Choose np.float32 or np.float64")

        super().__init__()
        self._filename = str(filename)
        input = self._filename if dfs is None else dfs
        self._read_header(input)
        self._dtype = dtype

        # show progress bar for large files
        # if self._type == UnstructuredType.Mesh:
        #    tot_size = self.n_elements
        # else:
        #    tot_size = self.n_elements * self.n_timesteps * self.n_items
        # if tot_size > 1e6:
        #    self.show_progress = True

    @property
    def deletevalue(self):
        """File delete value"""
        return self._deletevalue

    @property
    def n_items(self):
        """Number of items"""
        return len(self.items)

    @property
    def items(self):
        """List of items"""
        return self._items

    @property
    def start_time(self):
        """File start time"""
        return self._start_time

    @property
    def n_timesteps(self):
        """Number of time steps"""
        return self._n_timesteps

    @property
    def timestep(self):
        """Time step size in seconds"""
        return self._timestep_in_seconds

    @property
    def end_time(self):
        """File end time"""
        return self.start_time + timedelta(
            seconds=((self.n_timesteps - 1) * self.timestep)
        )

    def read(
        self,
        *,
        items=None,
        time=None,
        time_steps=None,
        elements=None,
        area=None,
        x=None,
        y=None,
    ) -> Dataset:
        """
        Read data from a dfsu file

        Parameters
        ---------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time: str, int or list[int], optional
            Read only selected time_steps
        area: list[float], optional
            Read only data inside (horizontal) area given as a
            bounding box (tuple with left, lower, right, upper)
            or as list of coordinates for a polygon, by default None
        x, y: float, optional
            Read only data for elements containing the (x,y) points(s),
            by default None
        elements: list[int], optional
            Read only selected element ids, by default None

        Returns
        -------
        Dataset
            A Dataset with data dimensions [t,elements]

        Examples
        --------
        >>> dfsu.read()
        <mikeio.Dataset>
        Dimensions: (9, 884)
        Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00
        Items:
        0:  Surface elevation <Surface Elevation> (meter)
        1:  U velocity <u velocity component> (meter per sec)
        2:  V velocity <v velocity component> (meter per sec)
        3:  Current speed <Current Speed> (meter per sec)
        >>> dfsu.read(time_steps="1985-08-06 12:00,1985-08-07 00:00")
        <mikeio.Dataset>
        Dimensions: (5, 884)
        Time: 1985-08-06 12:00:00 - 1985-08-06 22:00:00
        Items:
        0:  Surface elevation <Surface Elevation> (meter)
        1:  U velocity <u velocity component> (meter per sec)
        2:  V velocity <v velocity component> (meter per sec)
        3:  Current speed <Current Speed> (meter per sec)
        """

        # Open the dfs file for reading
        # self._read_dfsu_header(self._filename)
        dfs = DfsuFile.Open(self._filename)
        # time may have changes since we read the header
        # (if engine is continuously writing to this file)
        # TODO: add more checks that this is actually still the same file
        # (could have been replaced in the meantime)

        self._n_timesteps = dfs.NumberOfTimeSteps
        if time_steps is not None:
            warnings.warn(
                FutureWarning(
                    "time_steps have been renamed to time, and will be removed in a future release"
                )
            )
            time = time_steps

        single_time_selected = np.isscalar(time) if time is not None else False
        time_steps = _valid_timesteps(dfs, time)

        self._validate_elements_and_geometry_sel(elements, area=area, x=x, y=y)
        if elements is None:
            elements = self._parse_geometry_sel(area=area, x=x, y=y)

        if elements is None:
            geometry = self.geometry
            n_elems = geometry.n_elements
        else:
            elements = [elements] if np.isscalar(elements) else elements
            n_elems = len(elements)
            geometry = self.geometry.elements_to_geometry(elements)

        item_numbers = _valid_item_numbers(
            dfs.ItemInfo, items, ignore_first=self.is_layered
        )
        items = _get_item_info(dfs.ItemInfo, item_numbers, ignore_first=self.is_layered)
        n_items = len(item_numbers)

        deletevalue = self.deletevalue

        data_list = []

        n_steps = len(time_steps)
        shape = (n_elems,) if single_time_selected else (n_steps, n_elems)
        for item in range(n_items):
            # Initialize an empty data block
            data = np.ndarray(shape=shape, dtype=self._dtype)
            data_list.append(data)

        t_seconds = np.zeros(n_steps, dtype=float)

        for i in trange(n_steps, disable=not self.show_progress):
            it = time_steps[i]
            for item in range(n_items):

                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)
                d = itemdata.Data
                d[d == deletevalue] = np.nan

                if elements is not None:
                    d = d[elements]

                if single_time_selected:
                    data_list[item] = d
                else:
                    data_list[item][i] = d

            t_seconds[i] = itemdata.Time

        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)

        dfs.Close()

        dims = ("time", "element") if not single_time_selected else ("element",)

        if elements is not None and len(elements) == 1:
            # squeeze point data
            dims = tuple([d for d in dims if d != "element"])
            data_list = [np.squeeze(d) for d in data_list]

        return Dataset(
            data_list, time, items, geometry=geometry, dims=dims, validate=False
        )

    def _validate_elements_and_geometry_sel(self, elements, **kwargs):
        used_kwargs = []
        for kw, val in kwargs.items():
            if val is not None:
                used_kwargs.append(kw)

        if elements is not None:
            for kw in used_kwargs:
                raise ValueError(f"Cannot select both {kw} and elements!")

        if "area" in used_kwargs and ("x" in used_kwargs or "y" in used_kwargs):
            raise ValueError(f"Cannot select both x,y and area!")

    def _parse_geometry_sel(self, area, x, y):
        elements = None

        if area is not None:
            elements = self.geometry._elements_in_area(area)

        if (x is not None) or (y is not None):
            elements = self.geometry.find_index(x=x, y=y)

        return elements

    def write_header(
        self,
        filename,
        start_time=None,
        dt=None,
        items=None,
        elements=None,
        title=None,
    ):
        """Write the header of a new dfsu file (for writing huge files)

        Parameters
        -----------
        filename: str
            full path to the new dfsu file
        start_time: datetime, optional
            start datetime, default is datetime.now()
        dt: float, optional
            The time step (in seconds)
        items: list[ItemInfo], optional
        elements: list[int], optional
            write only these element ids to file
        title: str
            title of the dfsu file. Default is blank.

        Examples
        --------
        >>> msh = Mesh("foo.mesh")
        >>> n_elements = msh.n_elements
        >>> dfs = Dfsu(meshfilename)
        >>> nt = 1000
        >>> n_items = 10
        >>> items = [ItemInfo(f"Item {i+1}") for i in range(n_items)]
        >>> with dfs.write_header(outfilename, items=items) as f:
        >>>     for i in range(1, nt):
        >>>         data = []
        >>>         for i in range(n_items):
        >>>             d = np.random.random((1, n_elements))
        >>>             data.append(d)
        >>>             f.append(data)
        """

        return self.write(
            filename=filename,
            data=[],
            start_time=start_time,
            dt=dt,
            items=items,
            elements=elements,
            title=title,
            keep_open=True,
        )

    def write(
        self,
        filename,
        data,
        start_time=None,
        dt=None,
        items=None,
        elements=None,
        title=None,
        keep_open=False,
    ):
        """Write a new dfsu file

        Parameters
        -----------
        filename: str
            full path to the new dfsu file
        data: list[np.array] or Dataset
            list of matrices, one for each item. Matrix dimension: time, x
        start_time: datetime, optional, deprecated
            start datetime, default is datetime.now()
        dt: float, optional, deprecated
            The time step (in seconds)
        items: list[ItemInfo], optional, deprecated
        elements: list[int], optional
            write only these element ids to file
        title: str
            title of the dfsu file. Default is blank.
        keep_open: bool, optional
            Keep file open for appending
        """
        if self.is_spectral:
            raise ValueError("write() is not supported for spectral dfsu!")

        if start_time:
            warnings.warn(
                "setting start_time is deprecated, please supply data in the form of a Dataset",
                FutureWarning,
            )

        if dt:
            warnings.warn(
                "setting dt is deprecated, please supply data in the form of a Dataset",
                FutureWarning,
            )

        if items:
            warnings.warn(
                "setting items is deprecated, please supply data in the form of a Dataset",
                FutureWarning,
            )

        if isinstance(data, list):
            warnings.warn(
                "supplying data as a list of numpy arrays is deprecated, please supply data in the form of a Dataset",
                FutureWarning,
            )
            if self.is_layered:
                raise ValueError(
                    "Layered dfsu files can only be written by providing a Dataset"
                )

        filename = str(filename)

        if isinstance(data, Dataset):
            items = data.items
            start_time = data.time[0]
            if dt is None and len(data.time) > 1:
                if not data.is_equidistant:
                    raise Exception(
                        "Data is not equidistant in time. Dfsu requires equidistant temporal axis!"
                    )
                dt = (data.time[1] - data.time[0]).total_seconds()
            if data.geometry.is_layered:
                zn_dynamic = data[0]._zn

            # data needs to be a list so we can fit zn later
            data = [x.to_numpy() for x in data]

        n_items = len(data)
        n_time_steps = 0
        if n_items > 0:
            n_time_steps = np.shape(data[0])[0]

        if dt is None:
            if self.timestep is None:
                dt = 1
            else:
                dt = self.timestep  # 1 # Arbitrary if there is only a single timestep

        if start_time is None:
            if self.start_time is None:
                start_time = datetime.now()
                warnings.warn(
                    f"No start time supplied. Using current time: {start_time} as start time."
                )
            else:
                start_time = self.start_time
                warnings.warn(
                    f"No start time supplied. Using start time from source: {start_time} as start time."
                )

        if items is None:
            if n_items == 0:
                raise ValueError(
                    "Number of items unknown. Add (..., items=[ItemInfo(...)]"
                )
            items = [ItemInfo(f"Item {i + 1}") for i in range(n_items)]

        if title is None:
            title = ""

        file_start_time = start_time

        # spatial subset
        if elements is None:
            geometry = self.geometry
        else:
            geometry = self.geometry.elements_to_geometry(elements)
            if (not self.is_2d) and (geometry._type == DfsuFileType.Dfsu2D):
                # redo extraction as 2d:
                # print("will redo extraction in 2d!")
                geometry = self.geometry.elements_to_geometry(
                    elements, node_layers="bottom"
                )
                if (items[0].name == "Z coordinate") and (
                    items[0].type == EUMType.ItemGeometry3D
                ):
                    # get rid of z-item
                    items = items[1 : (n_items + 1)]
                    n_items = n_items - 1
                    new_data = []
                    for j in range(n_items):
                        new_data.append(data[j + 1])
                    data = new_data

        if geometry.is_layered:
            z_item = ItemInfo(
                "Z coordinate", itemtype=EUMType.ItemGeometry3D, unit=EUMUnit.meter
            )
            items.insert(0, z_item)
            n_items = len(items)
            data.insert(0, zn_dynamic)

        # Default filetype;
        if geometry._type is None:  # == DfsuFileType.Mesh:
            # create dfs2d from mesh
            dfsu_filetype = DfsuFileType.Dfsu2D
        else:
            #    # TODO: if subset is slice...
            dfsu_filetype = geometry._type.value

        if dfsu_filetype != DfsuFileType.Dfsu2D:
            if (items[0].name != "Z coordinate") and (
                items[0].type == EUMType.ItemGeometry3D
            ):
                raise Exception("First item must be z coordinates of the nodes!")

        xn = geometry.node_coordinates[:, 0]
        yn = geometry.node_coordinates[:, 1]

        # zn have to be Single precision??
        zn = geometry.node_coordinates[:, 2]

        # TODO verify this
        # elem_table = geometry.element_table
        elem_table = []
        for j in range(geometry.n_elements):
            elem_nodes = geometry.element_table[j]
            elem_nodes = [nd + 1 for nd in elem_nodes]
            elem_table.append(np.array(elem_nodes))
        elem_table = elem_table

        builder = DfsuBuilder.Create(dfsu_filetype)

        builder.SetNodes(xn, yn, zn, geometry.codes)
        builder.SetElements(elem_table)
        # builder.SetNodeIds(geometry.node_ids+1)
        # builder.SetElementIds(geometry.elements+1)

        factory = DfsFactory()
        proj = factory.CreateProjection(geometry.projection_string)
        builder.SetProjection(proj)
        builder.SetTimeInfo(file_start_time, dt)
        builder.SetZUnit(eumUnit.eumUmeter)

        if dfsu_filetype != DfsuFileType.Dfsu2D:
            builder.SetNumberOfSigmaLayers(geometry.n_sigma_layers)

        for item in items:
            if item.name != "Z coordinate":
                builder.AddDynamicItem(
                    item.name, eumQuantity.Create(item.type, item.unit)
                )

        try:
            self._dfs = builder.CreateFile(filename)
        except IOError:
            print("cannot create dfsu file: ", filename)

        deletevalue = self._dfs.DeleteValueFloat

        try:
            # Add data for all item-timesteps, copying from source
            for i in trange(n_time_steps, disable=not self.show_progress):
                for item in range(len(items)):
                    d = data[item][i, :]
                    d[np.isnan(d)] = deletevalue
                    darray = d
                    self._dfs.WriteItemTimeStepNext(0, darray.astype(np.float32))
            if not keep_open:
                self._dfs.Close()
            else:
                return self

        except Exception as e:
            print(e)
            self._dfs.Close()
            os.remove(filename)

    def append(self, data: Dataset) -> None:
        """Append to a dfsu file opened with `write(...,keep_open=True)`

        Parameters
        -----------
        data: Dataset
        """

        deletevalue = self._dfs.DeleteValueFloat
        n_items = len(data)
        n_time_steps = np.shape(data[0])[0]
        for i in trange(n_time_steps, disable=not self.show_progress):
            for item in range(n_items):
                di = data[item]
                if isinstance(data, Dataset):
                    di = di.to_numpy()
                d = di[i, :]
                d[np.isnan(d)] = deletevalue
                darray = d.astype(np.float32)
                self._dfs.WriteItemTimeStepNext(0, darray)

    def close(self):
        "Finalize write for a dfsu file opened with `write(...,keep_open=True)`"
        self._dfs.Close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._dfs.Close()

    def to_mesh(self, outfilename):
        """write object to mesh file

        Parameters
        ----------
        outfilename : str
            path to file to be written
        """
        if self.is_2d:
            # make sure element table has been constructured
            _ = self.element_table
            geometry = self.geometry
        else:
            geometry = self._geometry2d

        geometry.to_mesh(outfilename)


class Dfsu2DH(_Dfsu):
    def extract_track(self, track, items=None, method="nearest"):
        """
        Extract track data from a dfsu file

        Parameters
        ---------
        track: pandas.DataFrame
            with DatetimeIndex and (x, y) of track points as first two columns
            x,y coordinates must be in same coordinate system as dfsu
        track: str
            filename of csv or dfs0 file containing t,x,y
        items: list[int] or list[str], optional
            Extract only selected items, by number (0-based), or by name
        method: str, optional
            Spatial interpolation method ('nearest' or 'inverse_distance')
            default='nearest'

        Returns
        -------
        Dataset
            A dataset with data dimension t
            The first two items will be x- and y- coordinates of track

        Examples
        --------
        >>> ds = dfsu.extract_track(times, xy, items=['u','v'])

        >>> ds = dfsu.extract_track('track_file.dfs0')

        >>> ds = dfsu.extract_track('track_file.csv', items=0)
        """
        if self.is_spectral:
            raise ValueError("Method not supported for spectral dfsu!")

        dfs = DfsuFile.Open(self._filename)

        item_numbers = _valid_item_numbers(dfs.ItemInfo, items)
        items = _get_item_info(dfs.ItemInfo, item_numbers)
        n_items = len(item_numbers)

        self._n_timesteps = dfs.NumberOfTimeSteps
        time_steps = _valid_timesteps(dfs, time_steps=None)

        deletevalue = self.deletevalue

        if isinstance(track, str):
            filename = track
            if os.path.exists(filename):
                _, ext = os.path.splitext(filename)
                if ext == ".dfs0":
                    df = Dfs0(filename).to_dataframe()
                elif ext == ".csv":
                    df = pd.read_csv(filename, index_col=0, parse_dates=True)
                else:
                    raise ValueError(f"{ext} files not supported (dfs0, csv)")

                times = df.index
                coords = df.iloc[:, 0:2].to_numpy(copy=True)
            else:
                raise ValueError(f"{filename} does not exist")
        elif isinstance(track, Dataset):
            times = track.time
            coords = np.zeros(shape=(len(times), 2))
            coords[:, 0] = track[0].to_numpy().copy()
            coords[:, 1] = track[1].to_numpy().copy()
        else:
            assert isinstance(track, pd.DataFrame)
            times = track.index
            coords = track.iloc[:, 0:2].to_numpy(copy=True)

        assert isinstance(
            times, pd.DatetimeIndex
        ), "The index must be a pandas.DatetimeIndex"
        assert (
            times.is_monotonic_increasing
        ), "The time index must be monotonic increasing. Consider df.sort_index() before passing to extract_track()."

        data_list = []
        data_list.append(coords[:, 0])  # longitude
        data_list.append(coords[:, 1])  # latitude
        for item in range(n_items):
            # Initialize an empty data block
            data = np.empty(shape=(len(times)), dtype=self._dtype)
            data[:] = np.nan
            data_list.append(data)

        if self.is_geo:
            lon = coords[:, 0]
            lon[lon < -180] = lon[lon < -180] + 360
            lon[lon >= 180] = lon[lon >= 180] - 360
            coords[:, 0] = lon

        # track end (relative to dfsu)
        t_rel = (times - self.end_time).total_seconds()
        # largest idx for which (times - self.end_time)<=0
        tmp = np.where(t_rel <= 0)[0]
        if len(tmp) == 0:
            raise ValueError("No time overlap! Track ends before dfsu starts!")
        i_end = tmp[-1]

        # track time relative to dfsu start
        t_rel = (times - self.start_time).total_seconds()
        tmp = np.where(t_rel >= 0)[0]
        if len(tmp) == 0:
            raise ValueError("No time overlap! Track starts after dfsu ends!")
        i_start = tmp[0]  # smallest idx for which t_rel>=0

        dfsu_step = int(np.floor(t_rel[i_start] / self.timestep))  # first step

        # spatial interpolation
        n_pts = 1 if method == "nearest" else 5
        elem_ids, weights = self.geometry.get_2d_interpolant(
            coords[i_start : (i_end + 1)], n_nearest=n_pts
        )

        # initialize dfsu data arrays
        d1 = np.ndarray(shape=(n_items, self.n_elements), dtype=self._dtype)
        d2 = np.ndarray(shape=(n_items, self.n_elements), dtype=self._dtype)
        t1 = 0.0
        t2 = 0.0

        # very first dfsu time step
        step = time_steps[dfsu_step]
        for item in range(n_items):
            itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, step)
            t2 = itemdata.Time - 1e-10
            # d = to_numpy(itemdata.Data)
            d = itemdata.Data
            d[d == deletevalue] = np.nan
            d2[item, :] = d

        def is_EOF(step):
            return step >= self.n_timesteps

        # loop over track points
        for i_interp, i in enumerate(
            trange(i_start, i_end + 1, disable=not self.show_progress)
        ):
            t_rel[i]  # time of point relative to dfsu start

            read_next = t_rel[i] > t2

            while (read_next == True) and (not is_EOF(dfsu_step + 1)):
                dfsu_step = dfsu_step + 1

                # swap new to old
                d1, d2 = d2, d1
                t1, t2 = t2, t1

                step = time_steps[dfsu_step]
                for item in range(n_items):
                    itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, step)
                    t2 = itemdata.Time
                    # d = to_numpy(itemdata.Data)
                    d = itemdata.Data
                    d[d == deletevalue] = np.nan
                    d2[item, :] = d

                read_next = t_rel[i] > t2

            if (read_next == True) and (is_EOF(dfsu_step)):
                # cannot read next - no more timesteps in dfsu file
                continue

            w = (t_rel[i] - t1) / self.timestep  # time-weight
            eid = elem_ids[i_interp]
            if np.any(eid > 0):
                dati = (1 - w) * np.dot(d1[:, eid], weights[i_interp])
                dati = dati + w * np.dot(d2[:, eid], weights[i_interp])
            else:
                dati = np.empty(shape=n_items, dtype=self._dtype)
                dati[:] = np.nan

            for item in range(n_items):
                data_list[item + 2][i] = dati[item]

        dfs.Close()

        items_out = []
        if self.is_geo:
            items_out.append(ItemInfo("Longitude"))
            items_out.append(ItemInfo("Latitude"))
        else:
            items_out.append(ItemInfo("x"))
            items_out.append(ItemInfo("y"))
        for item in items:
            items_out.append(item)

        return Dataset(data_list, times, items_out)


class Mesh(_UnstructuredFile):
    """
    The Mesh class is initialized with a mesh or a dfsu file.

    Parameters
    ---------
    filename: str
        dfsu or mesh filename

    Examples
    --------

    >>> msh = Mesh("../tests/testdata/odense_rough.mesh")
    >>> msh
    Number of elements: 654
    Number of nodes: 399
    Projection: UTM-33

    """

    def __init__(self, filename):
        super().__init__()
        self._filename = filename
        self._read_header(filename)
        self._n_timesteps = None
        self._type = None  # DfsuFileType.Mesh
        self.plot = self.geometry.plot

    @property
    def zn(self):
        """Static bathymetry values (depth) at nodes"""
        return self.node_coordinates[:, 2]

    @zn.setter
    def zn(self, v):
        if len(v) != self.n_nodes:
            raise ValueError(f"zn must have length of nodes ({self.n_nodes})")
        self._geometry._nc[:, 2] = v
        self._geometry._ec = None

    def write(self, outfilename, elements=None):
        """write mesh to file (will overwrite if file exists)

        Parameters
        ----------
        outfilename : str
            path to file
        elements : list(int)
            list of element ids (subset) to be saved to new mesh
        """
        builder = MeshBuilder()

        if elements is None:
            geometry = self.geometry
            if hasattr(self._source, "EumQuantity"):
                quantity = self._source.EumQuantity
            else:
                quantity = eumQuantity.Create(EUMType.Bathymetry, self._source.ZUnit)
            elem_table = self._source.ElementTable
        else:
            geometry = self.geometry.elements_to_geometry(elements)
            quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)
            elem_table = _UnstructuredFile._element_table_to_mikecore(
                geometry.element_table
            )

        nc = geometry.node_coordinates
        builder.SetNodes(nc[:, 0], nc[:, 1], nc[:, 2], geometry.codes)

        builder.SetElements(elem_table)
        builder.SetProjection(geometry.projection_string)
        builder.SetEumQuantity(quantity)

        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)

    # @wraps(GeometryFM.plot.boundary_nodes)
    def plot_boundary_nodes(self, boundary_names=None, figsize=None, ax=None):
        return self.geometry.plot.boundary_nodes(boundary_names, figsize, ax)

    # @staticmethod
    # def _geometry_to_mesh(outfilename, geometry):

    #     builder = MeshBuilder()

    #     nc = geometry.node_coordinates
    #     builder.SetNodes(nc[:, 0], nc[:, 1], nc[:, 2], geometry.codes)
    #     # builder.SetNodeIds(geometry.node_ids+1)
    #     # builder.SetElementIds(geometry.elements+1)
    #     builder.SetElements(
    #         _UnstructuredFile._element_table_to_mikecore(geometry.element_table)
    #     )
    #     builder.SetProjection(geometry.projection_string)
    #     quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)
    #     builder.SetEumQuantity(quantity)
    #     newMesh = builder.CreateMesh()
    #     newMesh.Write(outfilename)
