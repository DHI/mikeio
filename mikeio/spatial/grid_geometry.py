from typing import Tuple, Union
import warnings
import numpy as np
from mikecore.eum import eumQuantity
from mikecore.MeshBuilder import MeshBuilder
from .geometry import (
    _Geometry,
    GeometryPoint2D,
    GeometryPoint3D,
    GeometryUndefined,
    BoundingBox,
)
from ..eum import EUMType, EUMUnit


def _check_equidistant(x: np.ndarray) -> None:
    d = np.diff(x)
    if len(d) > 0 and not np.allclose(d, d[0]):
        raise NotImplementedError("values must be equidistant")


def _parse_grid_axis(name, x, x0=0.0, dx=None, nx=None):
    if x is not None:
        x = np.asarray(x)
        _check_equidistant(x)
        if len(x) > 1 and x[0] > x[-1]:
            raise ValueError("{name} values must be increasing")
        x0 = x[0]
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        nx = len(x)
    else:
        if nx is None:
            raise ValueError(f"n{name} must be provided")
        if dx is None:
            raise ValueError(f"d{name} must be provided")
        if dx <= 0:
            raise ValueError(f"d{name} must be positive")
    return x0, dx, nx


class Grid1D(_Geometry):
    """1D grid (node-based)
    axis is increasing and equidistant
    """

    def __init__(
        self,
        x=None,
        *,
        x0=0.0,
        dx=None,
        nx=None,
        projection="NON-UTM",
        origin: Tuple[float, float] = (0.0, 0.0),
        orientation=0.0,
        node_coordinates=None,
        axis_name="x",
    ):
        """Create equidistant 1D spatial geometry"""
        self._projstr = projection  # TODO handle other types than string
        self._origin = origin
        self._orientation = orientation
        self._x0, self._dx, self._nx = _parse_grid_axis("x", x, x0, dx, nx)

        if node_coordinates is not None and len(node_coordinates) != self.nx:
            raise ValueError("Length of node_coordinates must be n")
        self._nc = node_coordinates

        self._axis_name = axis_name

    def __repr__(self):
        out = []
        out.append("<mikeio.Grid1D>")
        out.append(
            f"axis: nx={self.nx} points from x0={self.x0:g} to x1={self.x1:g} with dx={self.dx:g}"
        )
        return str.join("\n", out)

    def __str__(self):
        return f"Grid1D (n={self.nx}, dx={self.dx:.4g})"

    def find_index(self, x: float, **kwargs) -> int:

        d = (self.x - x) ** 2
        return np.argmin(d)

    def get_spatial_interpolant(self, xy, **kwargs):

        x = xy[0][0]  # TODO accept list of points
        d = np.abs(self.x - x)
        ids = np.argsort(d)[0:2]
        weights = 1 - d[ids]

        assert np.allclose(weights.sum(), 1.0)
        assert len(ids) == 2
        return ids, weights

    def interp(self, data, ids, weights):
        return np.dot(data[:, ids], weights)

    @property
    def dx(self) -> float:
        """grid spacing"""
        return self._dx

    @property
    def x(self):
        """array of node coordinates"""
        x1 = self.x0 + self.dx * (self.nx - 1)
        return np.linspace(self.x0, x1, self.nx)

    @property
    def x0(self) -> float:
        """left end-point"""
        return self._x0

    @property
    def x1(self) -> float:
        """right end-point"""
        return self.x[-1]

    @property
    def nx(self) -> int:
        """number of grid points"""
        return self._nx

    @property
    def origin(self) -> Tuple[float, float]:
        return self._origin

    @property
    def orientation(self) -> float:
        return self._orientation

    def isel(self, idx, axis=0):

        if not np.isscalar(idx):
            nc = None if self._nc is None else self._nc[idx, :]
            return Grid1D(
                x=self.x[idx],
                projection=self.projection,
                origin=self.origin,
                orientation=self.orientation,
                node_coordinates=nc,
            )

        if self._nc is None:
            return GeometryUndefined()
        else:
            coords = self._nc[idx, :]
            if len(coords) == 3:
                return GeometryPoint3D(*coords)
            else:
                return GeometryPoint2D(*coords)


class Grid2D(_Geometry):
    """2D grid
    Origin in the center of cell in lower-left corner
    x and y axes are increasing and equidistant
    """

    def __init__(
        self,
        *,
        x=None,
        x0=0.0,
        dx=None,
        nx=None,
        y=None,
        y0=0.0,
        dy=None,
        ny=None,
        bbox=None,
        projection="NON-UTM",
        origin: Tuple[float, float] = None,
        orientation=0.0,
        axis_names=("x", "y"),
        is_spectral=False,
    ):
        """Create equidistant 1D spatial geometry"""
        self._projstr = projection  # TODO handle other types than string
        self._shift_origin_on_write = origin is None  # user-constructed
        self._origin = (0.0, 0.0) if origin is None else origin
        self._orientation = orientation
        self.__xx = None
        self.__yy = None

        self._axis_names = axis_names

        if bbox is not None:
            if (x0 != 0.0) or (y0 != 0.0):
                raise ValueError("x0,y0 cannot be provided together with bbox")
            self._create_in_bbox(bbox, dx=dx, dy=dy, nx=nx, ny=ny)
        else:
            self._x0, self._dx, self._nx = _parse_grid_axis("x", x, x0, dx, nx)
            dy = self._dx if dy is None else dy
            self._y0, self._dy, self._ny = _parse_grid_axis("y", y, y0, dy, ny)

        self._x_logarithmic = False
        self._is_spectral = is_spectral

    @property
    def is_spectral(self):
        return self._is_spectral

    @is_spectral.setter
    def is_spectral(self, value):
        if self._is_spectral and (not value):
            self._is_spectral = False
            self._x_logarithmic = False
        if (not self._is_spectral) and value:
            self._is_spectral = True
            self._x_logarithmic = self.dx > 1.0

    @property
    def _is_rotated(self):
        return np.abs(self._orientation) > 1e-5

    def _create_in_bbox(self, bbox, dx=None, dy=None, nx=None, ny=None):
        """create 2d grid in bounding box, specifying spacing or shape

        Parameters
        ----------
        bbox : array(float)
            [left, bottom, right, top] =
            [x0-dx/2, y0-dy/2, x1+dx/2, y1+dy/2]
        dx : float, optional
            grid resolution in x-direction,
            will overwrite left and right if given
        dy : float, optional
            grid resolution in y-direction,
            will overwrite bottom and top if given
        nx : int, optional
            number of points in x-direction can be None,
            in which case the value will be inferred
        ny : int, optional
            number of points in y-direction can be None,
            in which case the value will be inferred
        """
        left, bottom, right, top = self._parse_bbox(bbox)

        xr = right - left  # dx too large
        yr = top - bottom  # dy too large

        if (dx is None and dy is None) and (nx is None and ny is None):
            if xr <= yr:
                nx = 10
                ny = int(np.ceil(nx * yr / xr))
            else:
                ny = 10
                nx = int(np.ceil(ny * xr / yr))
        if nx is None and ny is not None:
            nx = int(np.ceil(ny * xr / yr))
        if ny is None and nx is not None:
            ny = int(np.ceil(nx * yr / xr))
        if isinstance(dx, tuple):
            dx, dy = dx
        dy = dx if dy is None else dy

        self._x0, self._dx, self._nx = self._create_in_bbox_1d("x", left, right, dx, nx)
        self._y0, self._dy, self._ny = self._create_in_bbox_1d("y", bottom, top, dy, ny)

    @staticmethod
    def _parse_bbox(bbox):
        left = bbox[0]
        bottom = bbox[1]
        right = bbox[2]
        top = bbox[3]

        if left > right:
            raise (
                ValueError(
                    f"Invalid x axis, left: {left} must be smaller than right: {right}"
                )
            )

        if bottom > top:
            raise (
                ValueError(
                    f"Invalid y axis, bottom: {bottom} must be smaller than top: {top}"
                )
            )
        return left, bottom, right, top

    @staticmethod
    def _create_in_bbox_1d(axname, left, right, dx=None, nx=None):
        xr = right - left
        if dx is not None:
            nx = int(np.ceil(xr / dx))

            # overwrite left and right! to make dx fit
            xcenter = left + xr / 2
            left = xcenter - dx * nx / 2
            right = xcenter + dx * nx / 2
        elif nx is not None:
            dx = xr / nx
        else:
            raise ValueError(f"Provide either d{axname} or n{axname}")
        x0 = left + dx / 2
        return x0, dx, nx

    def __repr__(self):
        out = []
        out.append("<mikeio.Grid2D>")
        out.append(
            f"axis: nx={self.nx} points from x0={self.x0:g} to x1={self.x1:g} with dx={self.dx:g}"
        )
        out.append(
            f"axis: ny={self.ny} points from y0={self.y0:g} to y1={self.y1:g} with dy={self.dy:g}"
        )
        return str.join("\n", out)

    def __str__(self):
        return f"Grid2D (ny={self.ny}, nx={self.nx})"

    @property
    def x0(self) -> float:
        return self._x0

    @property
    def y0(self) -> float:
        return self._y0

    @property
    def dx(self) -> float:
        """x grid spacing"""
        return self._dx

    @property
    def dy(self) -> float:
        """y grid spacing"""
        return self._dy

    @property
    def x(self):
        """array of x coordinates (element center)"""
        if self.is_spectral:
            return self.logarithmic_f(self.nx, self.x0, self.dx)

        x1 = self.x0 + self.dx * (self.nx - 1)
        x_local = np.linspace(self.x0, x1, self.nx)
        if self._is_rotated:
            return x_local
        else:
            return x_local + self._origin[0]

    @staticmethod
    def logarithmic_f(n=25, f0=0.055, freq_factor=1.1):
        """Generate logarithmic frequency axis

        Parameters
        ----------
        n : int, optional
            number of frequencies, by default 25
        f0 : float, optional
            Minimum frequency, by default 0.055
        freq_factor : float, optional
            Frequency factor, by default 1.1

        Returns
        -------
        np.ndarray
            array of logarithmic distributed discrete frequencies
        """
        logf0 = np.log(f0)
        logdf = np.log(f0 * freq_factor) - logf0
        logf = logf0 + logdf * np.arange(n)
        return np.exp(logf)

    @property
    def y(self):
        """array of y coordinates (element center)"""
        y1 = self.y0 + self.dy * (self.ny - 1)
        y_local = np.linspace(self.y0, y1, self.ny)
        return y_local if self._is_rotated else y_local + self._origin[1]

    @property
    def x0(self):
        """x starting point"""
        return self._x0

    @property
    def y0(self) -> float:
        """y starting point"""
        return self._y0

    @property
    def x1(self) -> float:
        """x end-point"""
        return self.x[-1]

    @property
    def y1(self) -> float:
        """y end-point"""
        return self.y[-1]

    @property
    def nx(self) -> int:
        """number of x grid points"""
        return self._nx

    @property
    def ny(self) -> int:
        """number of y grid points"""
        return self._ny

    @property
    def origin(self) -> Tuple[float, float]:
        return self._origin

    @property
    def orientation(self) -> float:
        return self._orientation

    @property
    def bbox(self):
        """bounding box (left, bottom, right, top)
        Note: not the same as the cell center values (x0,y0,x1,y1)!
        """
        if self._is_rotated:
            raise NotImplementedError("Only available if orientation = 0")
        if self.is_spectral:
            raise NotImplementedError("Not available for spectral Grid2D")
        left = self.x0 - self.dx / 2
        bottom = self.y0 - self.dy / 2
        right = self.x1 + self.dx / 2
        top = self.y1 + self.dy / 2
        return BoundingBox(left, bottom, right, top)

    @property
    def _xx(self):
        """2d array of all x-coordinates"""
        if self.__xx is None:
            self._create_meshgrid(self.x, self.y)
        return self.__xx

    @property
    def _yy(self):
        """2d array of all y-coordinates"""
        if self.__yy is None:
            self._create_meshgrid(self.x, self.y)
        return self.__yy

    def _create_meshgrid(self, x, y):
        self.__xx, self.__yy = np.meshgrid(x, y)

    @property
    def xy(self):
        """n-by-2 array of x- and y-coordinates"""
        xcol = self._xx.reshape(-1, 1)
        ycol = self._yy.reshape(-1, 1)
        return np.column_stack([xcol, ycol])

    @property
    def coordinates(self):
        # TODO: remove this?
        """n-by-2 array of x- and y-coordinates"""
        return self.xy

    def _shift_x0y0_to_origin(self):
        """Shift spatial axis to start at (0,0) adding the start to origin instead
        Note: this will note change the x or y properties.
        """
        if self._is_rotated:
            raise ValueError("Only possible if orientation = 0")
        if self.is_spectral:
            raise ValueError("Not possible for spectral Grid2D")
        x0, y0 = self.x0, self.y0
        self._x0, self._y0 = 0.0, 0.0
        self._origin = (self._origin[0] + x0, self._origin[1] + y0)

    def contains(self, xy):
        """test if a list of points are inside grid

        Parameters
        ----------
        xy : array(float)
            xy-coordinate of points given as n-by-2 array

        Returns
        -------
        bool array
            True for points inside, False otherwise
        """
        xy = np.atleast_2d(xy)
        y = xy[:, 1]
        x = xy[:, 0]

        xinside = (self.bbox.left <= x) & (x <= self.bbox.right)
        yinside = (self.bbox.bottom <= y) & (y <= self.bbox.top)
        return xinside & yinside

    # def find_index(self, x: float, y: float) -> Tuple[int, int]:

    #     dist_x = (self.x - x) ** 2
    #     idx_x = np.argmin(dist_x)
    #     dist_y = (self.y - y) ** 2
    #     idx_y = np.argmin(dist_y)
    #     return idx_x, idx_y

    def find_index(self, x: float = None, y: float = None, coords=None, area=None):
        """Find nearest index (i,j) of point(s)
           -1 is returned if point is outside grid

        Parameters
        ----------
        coords : array(float)
            xy-coordinate of points given as n-by-2 array
        area : array(float)
            xy-coordinates of bounding box

        Returns
        -------
        array(int), array(int)
            i- and j-index of nearest cell
        """
        if x is not None or y is not None:
            if coords is not None:
                raise ValueError("x,y and coords cannot be given at the same time!")
            if x is None or y is None:
                raise ValueError("please provide either both x AND y or coords!")
            coords = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])

        if coords is not None:
            return self._xy_to_index(coords)
        elif area is not None:
            return self._bbox_to_index(area)

    def _xy_to_index(self, xy):
        """Find specific points in this geometry"""
        xy = np.atleast_2d(xy)
        y = xy[:, 1]
        x = xy[:, 0]

        ii = (-1) * np.ones_like(x, dtype=int)
        jj = (-1) * np.ones_like(y, dtype=int)

        inside = self.contains(xy)
        for j, xyp in enumerate(xy):
            if inside[j]:
                ii[j] = (np.abs(self.x - xyp[0])).argmin()
                jj[j] = (np.abs(self.y - xyp[1])).argmin()

        return ii, jj

    def _bbox_to_index(self, bbox):
        """Find subarea within this geometry"""
        assert len(bbox) == 4, "area most be a bounding box of coordinates"
        x0, y0, x1, y1 = bbox
        if x0 > self.x1 or y0 > self.y1 or x1 < self.x0 or y1 < self.y0:
            warnings.warn("No elements in bbox")
            return None, None

        mask = (self.x >= x0) & (self.x <= x1)
        ii = np.where(mask)[0]
        mask = (self.y >= y0) & (self.y <= y1)
        jj = np.where(mask)[0]

        return ii, jj

    def isel(self, idx, axis):

        if not np.isscalar(idx):
            d = np.diff(idx)
            if np.any(d < 1) or not np.allclose(d, d[0]):
                return GeometryUndefined()
            else:
                x = self.x if axis == 0 else self.x[idx]
                y = self.y if axis == 1 else self.y[idx]
                return Grid2D(x=x, y=y, projection=self.projection)

        if axis == 0:
            # y is first axis! if we select an element from y-axis (axis 0),
            # we return a "copy" of the x-axis
            nc = np.column_stack([self.x, self.y[idx] * np.ones_like(self.x)])
            return Grid1D(x=self.x, projection=self.projection, node_coordinates=nc)
        else:
            nc = np.column_stack([self.x[idx] * np.ones_like(self.y), self.y])
            return Grid1D(x=self.y, projection=self.projection, node_coordinates=nc)

    def _index_to_geometry(self, ii, jj):
        di = np.diff(ii)
        dj = np.diff(jj)
        if (np.any(di < 1) or not np.allclose(di, di[0])) or (
            np.any(dj < 1) or not np.allclose(dj, dj[0])
        ):
            warnings.warn("Axis not equidistant! Will return GeometryUndefined()")
            return GeometryUndefined()
        else:
            return Grid2D(x=self.x[ii], y=self.x[jj], projection=self.projection)

    def _to_element_table(self, index_base=0):

        elem_table = []
        for elx in range(self.nx - 1):
            # each col
            for ely in range(self.ny - 1):
                # each row (in this col)
                n1 = ely * self.nx + elx + index_base
                n2 = (ely + 1) * self.nx + elx + index_base
                elem_table.append([n1, n1 + 1, n2 + 1, n2])
        return elem_table

    @staticmethod
    def _centers_to_nodes(x):
        """Nodes will be placed mid-way between centers
        If non-equidistant, new centers will hence not equal old centers!
        """
        if len(x) == 1:
            return np.array([x[0] - 0.5, x[0] + 0.5])
        xinner = (x[1:] + x[:-1]) / 2
        left = x[0] - (x[1] - x[0]) / 2
        right = x[-1] + (x[-1] - x[-2]) / 2
        return np.array([left, *xinner, right])

    @staticmethod
    def _nodes_to_centers(xn):
        return (xn[1:] + xn[:-1]) / 2

    def get_node_coordinates(self):
        """node coordinates for this grid

        Returns
        -------
        array(float)
            2d array with x,y-coordinates, length=(nx+1)*(ny+1)
        """
        xn = self._centers_to_nodes(self.x)
        yn = self._centers_to_nodes(self.y)
        gn = Grid2D(x=xn, y=yn)
        return gn.xy

    def to_mesh(self, outfilename, projection=None, z=None):
        """export grid to mesh file

        Parameters
        ----------
        outfilename : str
            path of new mesh file
        projection : str, optional
            WKT projection string, by default 'LONG/LAT'
        z : float or array(float), optional
            bathymetry values for each node, by default 0
            if array: must have length=(nx+1)*(ny+1)
        """
        if projection is None:
            projection = "LONG/LAT"

        # get node based grid
        xn = self._centers_to_nodes(self.x)
        yn = self._centers_to_nodes(self.y)
        gn = Grid2D(x=xn, y=yn)

        x = gn.xy[:, 0]
        y = gn.xy[:, 1]
        n = gn.nx * gn.ny
        if z is None:
            z = np.zeros(n)
        else:
            if np.isscalar(z):
                z = z * np.ones(n)
            else:
                if len(z) != n:
                    raise ValueError(
                        "z must either be scalar or have length of nodes ((nx+1)*(ny+1))"
                    )
        codes = np.zeros(n, dtype=int)
        codes[y == gn.bbox.top] = 5  # north
        codes[x == gn.bbox.right] = 4  # east
        codes[y == gn.bbox.bottom] = 3  # south
        codes[x == gn.bbox.left] = 2  # west
        codes[(y == gn.bbox.top) & (x == gn.bbox.left)] = 5  # corner->north

        builder = MeshBuilder()
        builder.SetNodes(x, y, z, codes)

        elem_table = gn._to_element_table(index_base=1)
        builder.SetElements(elem_table)

        builder.SetProjection(projection)
        quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)
        builder.SetEumQuantity(quantity)
        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)


class Grid3D(_Geometry):
    """3D  grid
    Origin in the center of cell in lower-left corner
    x, y and z axes are increasing and equidistant
    """

    def __init__(
        self,
        *,
        x=None,
        x0=0.0,
        dx=None,
        nx=None,
        y=None,
        y0=0.0,
        dy=None,
        ny=None,
        z=None,
        z0=0.0,
        dz=None,
        nz=None,
        projection="NON-UTM",
        origin: Tuple[float, float] = (0.0, 0.0),
        orientation=0.0,
    ) -> None:

        super().__init__()
        self._x0, self._dx, self._nx = _parse_grid_axis("x", x, x0, dx, nx)
        self._y0, self._dy, self._ny = _parse_grid_axis("y", y, y0, dy, ny)
        self._z0, self._dz, self._nz = _parse_grid_axis("z", z, z0, dz, nz)

        self._projstr = projection  # TODO handle other types than string
        self._origin = origin
        self._orientation = orientation

    @property
    def x(self):
        """array of x-axis node coordinates"""
        x1 = self.x0 + self.dx * (self.nx - 1)
        return np.linspace(self.x0, x1, self.nx)

    @property
    def x0(self) -> float:
        return self._x0

    @property
    def y0(self) -> float:
        return self._y0

    @property
    def z0(self) -> float:
        return self._z0

    @property
    def dx(self) -> float:
        """x-axis grid spacing"""
        return self._dx

    @property
    def nx(self):
        """number of x-axis nodes"""
        return self._nx

    @property
    def y(self):
        """array of y-axis node coordinates"""
        y1 = self.y0 + self.dy * (self.ny - 1)
        return np.linspace(self.y0, y1, self.ny)

    @property
    def dy(self) -> float:
        """y-axis grid spacing"""
        return self._dy

    @property
    def ny(self):
        """number of y-axis nodes"""
        return self._ny

    @property
    def z(self):
        """array of z-axis node coordinates"""
        z1 = self.z0 + self.dz * (self.nz - 1)
        return np.linspace(self.z0, z1, self.nz)

    @property
    def dz(self) -> float:
        """z-axis grid spacing"""
        return self._dz

    @property
    def nz(self):
        """number of z-axis nodes"""
        return self._nz

    def find_index(self, coords=None, layer=None, area=None):
        if layer is not None:
            raise NotImplementedError(
                f"Layer slicing is not yet implemented. Use the mikeio.read('file.dfs3', layers='{layer}')"
            )
        raise NotImplementedError(
            "Not yet implemented for Grid3D. Please use mikeio.read('file.dfs3') and its arguments instead."
        )

    def isel(self, idx, axis):
        if not np.isscalar(idx):
            d = np.diff(idx)
            if np.any(d < 1) or not np.allclose(d, d[0]):
                return GeometryUndefined()
            else:
                x = self.x[idx] if axis == 2 else self.x
                y = self.y[idx] if axis == 1 else self.y
                z = self.z[idx] if axis == 0 else self.z
                return Grid3D(x=x, y=y, z=z, projection=self.projection)

        if axis == 0:
            # z is the first axis! return x-y Grid2D
            # TODO: origin, how to pass self.z[idx]?
            return Grid2D(
                x=self.x + self._origin[0],
                y=self.y + self._origin[1],
                projection=self.projection,
            )
        elif axis == 1:
            # y is the second axis! return x-z Grid2D
            # TODO: origin, how to pass self.y[idx]?
            return Grid2D(
                x=self.x,
                y=self.z,
                # projection=self._projection,
            )
        else:
            # x is the last axis! return y-z Grid2D
            # TODO: origin, how to pass self.x[idx]?
            return Grid2D(
                x=self.y,
                y=self.z,
                # projection=self._projection,
            )

    def __repr__(self):
        out = []
        out.append("<mikeio.Grid3D>")
        out.append(
            f"x-axis: nx={self.nx} points from x0={self.x[0]:g} to x1={self.x[-1]:g} with dx={self.dx:g}"
        )
        out.append(
            f"y-axis: ny={self.ny} points from y0={self.y[0]:g} to y1={self.y[-1]:g} with dy={self.dy:g}"
        )
        out.append(
            f"z-axis: nz={self.nz} points from z0={self.z[0]:g} to z1={self.z[-1]:g} with dz={self.dz:g}"
        )
        return str.join("\n", out)

    def __str__(self):
        return f"Grid3D(nz={self.nz}, ny={self.ny}, nx={self.nx})"

    def _geometry_for_layers(self, layers) -> Union[Grid2D, "Grid3D"]:
        if layers is None:
            return self

        g = self
        if len(layers) == 1:
            geometry = Grid2D(
                x=g.x + g._origin[0],
                y=g.y + g._origin[1],
                projection=g.projection,
            )
        else:
            d = np.diff(g.z[layers])
            if np.any(d < 1) or not np.allclose(d, d[0]):
                warnings.warn("Extracting non-equidistant layers! Cannot use Grid3D.")
                geometry = GeometryUndefined()
            else:
                geometry = Grid3D(
                    x=g.x,
                    y=g.y,
                    z=g.z[layers],
                    origin=g._origin,
                    projection=g.projection,
                )
        return geometry
