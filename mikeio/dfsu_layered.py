import numpy as np
import pandas as pd
import warnings
from tqdm import trange
from functools import wraps
from scipy.spatial import cKDTree

from mikecore.DfsuFile import DfsuFile, DfsuFileType
from .dfsu import _Dfsu
from .dataset import Dataset, DataArray
from .spatial.FM_geometry import GeometryFM3D
from .custom_exceptions import InvalidGeometry
from .dfsutil import _get_item_info, _valid_item_numbers, _valid_timesteps
from .spatial.FM_utils import _plot_vertical_profile
from .interpolation import get_idw_interpolant, interp2d
from .eum import ItemInfo, EUMType


class DfsuLayered(_Dfsu):
    @wraps(GeometryFM3D.to_2d_geometry)
    def to_2d_geometry(self):
        return self.geometry2d

    @property
    def geometry2d(self):
        """The 2d geometry for a 3d object"""
        return self._geometry2d

    @property
    def n_layers(self):
        """Maximum number of layers"""
        return self.geometry._n_layers

    @property
    def n_sigma_layers(self):
        """Number of sigma layers"""
        return self.geometry.n_sigma_layers

    @property
    def n_z_layers(self):
        """Maximum number of z-layers"""
        if self.n_layers is None:
            return None
        return self.n_layers - self.n_sigma_layers

    @property
    def e2_e3_table(self):
        """The 2d-to-3d element connectivity table for a 3d object"""
        if self.n_layers is None:
            print("Object has no layers: cannot return e2_e3_table")
            return None
        return self.geometry.e2_e3_table

    @property
    def elem2d_ids(self):
        """The associated 2d element id for each 3d element"""
        if self.n_layers is None:
            raise InvalidGeometry("Object has no layers: cannot return elem2d_ids")
        return self.geometry.elem2d_ids

    @property
    def layer_ids(self):
        """The layer number (0=bottom, 1, 2, ...) for each 3d element"""
        if self.n_layers is None:
            raise InvalidGeometry("Object has no layers: cannot return layer_ids")
        return self.geometry.layer_ids

    @property
    def top_elements(self):
        """List of 3d element ids of surface layer"""
        if self.n_layers is None:
            print("Object has no layers: cannot find top_elements")
            return None
        return self.geometry.top_elements

    @property
    def n_layers_per_column(self):
        """List of number of layers for each column"""
        if self.n_layers is None:
            print("Object has no layers: cannot find n_layers_per_column")
            return None
        return self.geometry.n_layers_per_column

    @property
    def bottom_elements(self):
        """List of 3d element ids of bottom layer"""
        if self.n_layers is None:
            print("Object has no layers: cannot find bottom_elements")
            return None
        return self.geometry.bottom_elements

    @wraps(GeometryFM3D.get_layer_elements)
    def get_layer_elements(self, layer):
        if self.n_layers is None:
            raise InvalidGeometry("Object has no layers: cannot get_layer_elements")
        return self.geometry.get_layer_elements(layer)

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
        z=None,
        layers=None,
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
        x, y, z: float, optional
            Read only data for elements containing the (x,y,z) points(s),
            by default None
        layers: int, str, list[int], optional
            Read only data for specific layers, by default None
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
        >>> dfsu.read(time="1985-08-06 12:00,1985-08-07 00:00")
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

        self._validate_elements_and_geometry_sel(
            elements, area=area, layers=layers, x=x, y=y, z=z
        )
        if elements is None:
            elements = self._parse_geometry_sel(area=area, layer=layers, x=x, y=y, z=z)

        if elements is None:
            n_elems = self.n_elements
            n_nodes = self.n_nodes
            geometry = self.geometry
        else:
            elements = [elements] if np.isscalar(elements) else elements
            n_elems = len(elements)
            geometry = self.geometry.elements_to_geometry(elements)
            if self.is_layered:  # and items[0].name == "Z coordinate":
                node_ids, _ = self.geometry._get_nodes_and_table_for_elements(elements)
                n_nodes = len(node_ids)

        item_numbers = _valid_item_numbers(
            dfs.ItemInfo, items, ignore_first=self.is_layered
        )
        items = _get_item_info(dfs.ItemInfo, item_numbers, ignore_first=self.is_layered)
        if self.is_layered:
            # we need the zn item too
            item_numbers = [it + 1 for it in item_numbers]
            if hasattr(geometry, "is_layered") and geometry.is_layered:
                item_numbers.insert(0, 0)
        n_items = len(item_numbers)

        deletevalue = self.deletevalue

        data_list = []

        n_steps = len(time_steps)
        item0_is_node_based = False
        for item in range(n_items):
            # Initialize an empty data block
            if hasattr(geometry, "is_layered") and geometry.is_layered and item == 0:
                # and items[item].name == "Z coordinate":
                item0_is_node_based = True
                data = np.ndarray(shape=(n_steps, n_nodes), dtype=self._dtype)
            else:
                data = np.ndarray(shape=(n_steps, n_elems), dtype=self._dtype)
            data_list.append(data)

        t_seconds = np.zeros(n_steps, dtype=float)

        if single_time_selected:
            data = data[0]

        for i in trange(n_steps, disable=not self.show_progress):
            it = time_steps[i]
            for item in range(n_items):

                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)
                d = itemdata.Data
                d[d == deletevalue] = np.nan

                if elements is not None:
                    if item == 0 and item0_is_node_based:
                        d = d[node_ids]
                    else:
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

        if hasattr(geometry, "is_layered") and geometry.is_layered:
            return Dataset(
                data_list[1:],  # skip zn item
                time,
                items,
                geometry=geometry,
                zn=data_list[0],
                dims=dims,
            )
        else:
            return Dataset(data_list, time, items, geometry=geometry, dims=dims)

    def _parse_geometry_sel(self, area, layer, x, y, z):
        elements = None
        if layer is not None:
            elements = self.geometry.get_layer_elements(layer)

        if area is not None:
            elements_a = self.geometry._elements_in_area(area)
            elements = (
                elements_a if layer is None else np.intersect1d(elements, elements_a)
            )

        if (x is not None) or (y is not None):
            elements_xy = self.geometry.find_index(x=x, y=y, z=z)
            elements = (
                elements_xy if layer is None else np.intersect1d(elements, elements_xy)
            )

        return elements


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
            cmin=cmin,
            cmax=cmax,
            label=label,
            **kwargs,
        )


class Dfsu3D(DfsuLayered):
    def find_nearest_profile_elements(self, x, y):
        """Find 3d elements of profile nearest to (x,y) coordinates

        Parameters
        ----------
        x : float
            x-coordinate of point
        y : float
            y-coordinate of point

        Returns
        -------
        np.array(int)
            element ids of vertical profile
        """
        if self.is_2d:
            raise InvalidGeometry("Object is 2d. Cannot get_nearest_profile")
        else:
            elem2d, _ = self.geometry._find_n_nearest_2d_elements(x, y)
            elem3d = self.geometry.e2_e3_table[elem2d]
            return elem3d

    def extract_surface_elevation_from_3d(self, filename=None, time=None, n_nearest=4):
        """
        Extract surface elevation from a 3d dfsu file (based on zn)
        to a new 2d dfsu file with a surface elevation item.

        Parameters
        ---------
        filename: str
            Output file name
        time: str, int or list[int], optional
            Extract only selected time_steps
        n_nearest: int, optional
            number of points for spatial interpolation (inverse_distance), default=4

        Examples
        --------
        >>> dfsu.extract_surface_elevation_from_3d('ex_surf.dfsu', time='2018-1-1,2018-2-1')
        """
        # validate input
        assert (
            self._type == DfsuFileType.Dfsu3DSigma
            or self._type == DfsuFileType.Dfsu3DSigmaZ
        )
        assert n_nearest > 0
        time_steps = _valid_timesteps(self._source, time)

        # make 2d nodes-to-elements interpolator
        top_el = self.top_elements
        geom = self.geometry.elements_to_geometry(top_el, node_layers="top")
        xye = geom.element_coordinates[:, 0:2]
        xyn = geom.node_coordinates[:, 0:2]
        tree2d = cKDTree(xyn)
        dist, node_ids = tree2d.query(xye, k=n_nearest)
        if n_nearest == 1:
            weights = None
        else:
            weights = get_idw_interpolant(dist)

        # read zn from 3d file and interpolate to element centers
        ds = self.read(items=0, time_steps=time_steps)  # read only zn
        node_ids_surf, _ = self.geometry._get_nodes_and_table_for_elements(
            top_el, node_layers="top"
        )
        zn_surf = ds[0].values[:, node_ids_surf]  # surface
        surf2d = interp2d(zn_surf, node_ids, weights)

        # create output
        items = [ItemInfo(EUMType.Surface_Elevation)]
        ds2 = Dataset([surf2d], ds.time, items, geometry=geom)
        if filename is None:
            return ds2
        else:
            title = "Surface extracted from 3D file"
            self.write(filename, ds2, elements=top_el, title=title)
