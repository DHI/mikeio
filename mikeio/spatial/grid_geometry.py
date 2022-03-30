from typing import Tuple
import numpy as np
from mikecore.eum import eumQuantity
from mikecore.MeshBuilder import MeshBuilder
from .geometry import _Geometry, GeometryUndefined, BoundingBox
from ..eum import EUMType, EUMUnit


class Grid1D(_Geometry):
    def __init__(
        self,
        x0=None,
        dx=None,
        n=None,
        projection="NON-UTM",
        origin: Tuple[float, float] = (0.0, 0.0),
        orientation=0.0,
    ):
        """Create equidistant 1D spatial geometry"""
        self._projection = projection
        self._projstr = projection  # TODO handle other types than string
        self._origin = origin
        self._orientation = orientation

        if n is None:
            raise ValueError("n must be provided")
        if dx is None:
            raise ValueError("dx must be provided")
        self._nx = n
        self._dx = dx
        self._x0 = 0 if x0 is None else x0
        x1 = self._x0 + dx * (n - 1)
        self._x = np.linspace(self._x0, x1, n)

    def __repr__(self):
        out = []
        out.append("<mikeio.Grid1D>")
        out.append(
            f"axis: nx={self.n} points from x0={self.x0:g} to x1={self.x1:g} with dx={self.dx:g}"
        )
        return str.join("\n", out)

    @property
    def _one_line_str(self) -> str:
        return f"Grid1D (n={self.n}, dx={self.dx:.4g})"

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
        return self._x

    @property
    def x0(self) -> float:
        """left end-point"""
        return self._x0

    @property
    def x1(self) -> float:
        """right end-point"""
        return self.x[-1]

    @property
    def n(self) -> int:
        """number of grid points"""
        return self._nx

    @property
    def origin(self) -> Tuple[float, float]:
        return self._origin

    @property
    def orientation(self) -> float:
        return self._orientation

    def isel(self, idx, axis):

        if not np.isscalar(idx):
            # TODO: return reduced Grid1D
            return None

        return GeometryUndefined()


class Grid2D(_Geometry):
    """2D horizontal grid
    Origin in the center of cell in lower-left corner
    x and y axes are increasing and equidistant
    """

    @property
    def x(self):
        """array of x-coordinates (single row)"""
        if self._x is None:
            self._x = self._create_axis(self._x0, self._dx, self._nx)
        return self._x

    @property
    def x0(self):
        """center of left end-point"""
        return self._x0

    @property
    def x1(self):
        """center of right end-point"""
        return self.x[-1]

    @property
    def dx(self):
        """x-spacing"""
        return self._dx

    @property
    def nx(self):
        """number of points in x-direction"""
        return self._nx

    @property
    def y(self):
        """array of y-coordinates (single column)"""
        if self._y is None:
            self._y = self._create_axis(self.y0, self.dy, self.ny)
        return self._y

    @property
    def y0(self):
        """center of lower end-point"""
        return self._y0

    @property
    def origin(self):
        return (self._x0, self._y0)

    @property
    def projection_string(self):
        return self._projection  # TODO

    @property
    def y1(self):
        """center of upper end-point"""
        return self.y[-1]

    @property
    def dy(self):
        """y-spacing"""
        return self._dy

    @property
    def ny(self):
        """number of cells in y-direction"""
        return self._ny

    @property
    def n(self):
        """total number of grid points"""
        return self._nx * self._ny

    @property
    def xx(self):
        """2d array of all x-coordinates"""
        if self._xx is None:
            self._create_meshgrid(self.x, self.y)
        return self._xx

    @property
    def yy(self):
        """2d array of all y-coordinates"""
        if self._yy is None:
            self._create_meshgrid(self.x, self.y)
        return self._yy

    @property
    def xy(self):
        """n-by-2 array of x- and y-coordinates"""
        xcol = self.xx.reshape(-1, 1)
        ycol = self.yy.reshape(-1, 1)
        return np.column_stack([xcol, ycol])

    @property
    def coordinates(self):
        """n-by-2 array of x- and y-coordinates"""
        return self.xy

    @property
    def bbox(self):
        """bounding box (left, bottom, right, top)
        Note: not the same as the cell center values (x0,y0,x1,y1)!
        """
        left = self.x0 - self.dx / 2
        bottom = self.y0 - self.dy / 2
        right = self.x1 + self.dx / 2
        top = self.y1 + self.dy / 2
        return BoundingBox(left, bottom, right, top)

    def __init__(
        self,
        x=None,
        y=None,
        bbox=None,
        dx=None,
        dy=None,
        shape=None,
        x0=None,
        y0=None,
        nx=None,
        ny=None,
        projection="LONG/LAT",
    ):
        """create 2d grid

        Parameters
        ----------
        x : array-like, optional
            1d array of x-coordinates (cell centers)
        y : array-like, optional
            1d array of y-coordinates (cell centers)
        bbox : array(float), optional
            [x0, y0, x1, y1]
        dx : float or (float, float), optional
            grid resolution in x-direction (or in x- and y-direction)
        dy : float, optional
            grid resolution in y-direction
        projection: str, optional
            default 'LONG/LAT'

        Examples
        --------
        >>> g = Grid2D(dx=0.25, nx=5, ny=10)
        >>> g = Grid2D(bbox=[0,0,10,20], dx=0.25)
        >>> g = Grid2D(bbox=[0,0,10,20], nx==5, ny=10)
        >>> x = np.linspace(0.0, 1000.0, 201)
        >>> y = [0, 2.0]
        >>> g = Grid2D(x, y)

        """
        self.orientation = 0.0
        self._projection = projection
        self._projstr = projection  # TODO handle other types than string

        self._x = None
        self._x0 = None
        self._dx = None
        self._nx = None

        self._y = None
        self._y0 = None
        self._dy = None
        self._ny = None

        self._xx = None
        self._yy = None

        dxdy = dx
        if dy is not None:
            if not np.isscalar(dx):
                dx = dx[0]

            if dx <= 0.0:
                raise ValueError("dx must be a positive number")
            if dy <= 0.0:
                raise ValueError("dy must be a positive number")
            dxdy = (dx, dy)

        if (x is not None) and (len(x) == 4):
            # first positional argument 'x' is probably bbox
            if (y is None) or ((dxdy is not None) or (shape is not None)):
                bbox, x = x, bbox

        if bbox is not None:
            self._create_in_bbox(bbox, dxdy, shape)
        elif (x is not None) and (y is not None):
            self._create_from_x_and_y(x, y)
        elif (nx is not None) and (ny is not None):
            self._create_from_nx_ny_dx_dy(x0=x0, dx=dx, nx=nx, y0=y0, dy=dy, ny=ny)
        else:
            raise ValueError(
                "Please provide either bbox or both x and y, or x0, dx, nx, y0, dy, ny"
            )

    def _create_in_bbox(self, bbox, dxdy=None, shape=None):
        """create 2d grid in bounding box, specifying spacing or shape

        Parameters
        ----------
        bbox : array(float)
            [x0, y0, x1, y1]
        dxdy : float or (float, float), optional
            grid resolution in x- and y-direction
        shape : (int, int), optional
            tuple with nx and ny describing number of points in each direction
            one of them can be None, in which case the value will be inferred
        """
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

        xr = right - left  # dx too large
        yr = top - bottom  # dy too large

        if (dxdy is None) and (shape is None):
            if xr <= yr:
                nx = 10
                ny = int(np.ceil(nx * yr / xr))
            else:
                ny = 10
                nx = int(np.ceil(ny * xr / yr))
            dx = xr / nx
            dy = yr / ny
        else:
            if shape is not None:
                if len(shape) != 2:
                    raise ValueError("shape must be (nx,ny)")
                nx, ny = shape
                if (nx is None) and (ny is None):
                    raise ValueError("nx and ny cannot both be None")
                if nx is None:
                    nx = int(np.ceil(ny * xr / yr))
                if ny is None:
                    ny = int(np.ceil(nx * yr / xr))
                dx = xr / nx
                dy = yr / ny
            elif dxdy is not None:
                if np.isscalar(dxdy):
                    dy = dx = dxdy
                else:
                    dx, dy = dxdy
                nx = int(np.ceil(xr / dx))
                ny = int(np.ceil(yr / dy))
            else:
                raise ValueError("dxdy and shape cannot both be provided! Chose one.")

        self._x0 = left + dx / 2
        self._dx = dx
        self._nx = nx

        self._y0 = bottom + dy / 2
        self._dy = dy
        self._ny = ny

    def _create_from_x_and_y(self, x, y):

        if x[0] > x[-1]:
            raise ValueError("x values must be increasing")

        if y[0] > y[-1]:
            raise ValueError("y values must be increasing")

        self._x0 = x[0]
        self._nx = len(x)
        self._dx = x[1] - x[0]

        self._y0 = y[0]
        self._ny = len(y)
        self._dy = y[1] - y[0]

    def _create_from_nx_ny_dx_dy(self, x0, dx, nx, y0, dy, ny):
        if nx is None:
            raise ValueError("nx must be provided")
        if dx is None:
            raise ValueError("dx must be provided")
        self._x0 = 0 if x0 is None else x0
        self._x = self._create_axis(self._x0, dx, nx)
        self._dx = dx
        self._nx = nx

        if ny is None:
            raise ValueError("ny must be provided")
        self._y0 = 0 if y0 is None else y0
        self._dy = dx if dy is None else dy
        self._y = self._create_axis(self._y0, self._dy, ny)
        self._ny = ny

    def _create_axis(self, x0, dx, nx):
        self._xx, self._yy = None, None
        x1 = x0 + dx * (nx - 1)
        return np.linspace(x0, x1, nx)

    def _create_meshgrid(self, x, y):
        self._xx, self._yy = np.meshgrid(x, y)

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

    def find_index(self, xy):
        """Find nearest index (i,j) of point(s)
           -1 is returned if point is outside grid

        Parameters
        ----------
        xy : array(float)
            xy-coordinate of points given as n-by-2 array

        Returns
        -------
        array(int), array(int)
            i- and j-index of nearest cell
        """

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

    def isel(self, idx, axis):

        if not np.isscalar(idx):
            # TODO: return reduced Grid2D
            return None

        if axis == 0:
            # y is first axis! if we select an element from y-axis (axis 0),
            # we return a "copy" of the x-axis
            return Grid1D(
                x0=self.x0,
                dx=self.dx,
                n=self.nx,
                projection=self._projection,
            )
        else:
            return Grid1D(
                x0=self.y0,
                dx=self.dy,
                n=self.ny,
                projection=self._projection,
            )

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
        xn = Grid2D._centers_to_nodes(self.x)
        yn = Grid2D._centers_to_nodes(self.y)
        gn = Grid2D(xn, yn)
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
        xn = Grid2D._centers_to_nodes(self.x)
        yn = Grid2D._centers_to_nodes(self.y)
        gn = Grid2D(xn, yn)

        x = gn.xy[:, 0]
        y = gn.xy[:, 1]
        if z is None:
            z = np.zeros(gn.n)
        else:
            if np.isscalar(z):
                z = z * np.ones(gn.n)
            else:
                if len(z) != gn.n:
                    raise ValueError(
                        "z must either be scalar or have length of nodes ((nx+1)*(ny+1))"
                    )
        codes = np.zeros(gn.n, dtype=int)
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

    @staticmethod
    def xy_to_bbox(xy, buffer=None):
        """return bounding box for list of coordinates"""
        if buffer is None:
            buffer = 0

        left = xy[:, 0].min() - buffer
        bottom = xy[:, 1].min() - buffer
        right = xy[:, 0].max() + buffer
        top = xy[:, 1].max() + buffer
        return BoundingBox(left, bottom, right, top)

    def __repr__(self):
        out = []
        out.append("<mikeio.Grid2D>")
        out.append(
            f"x-axis: nx={self.nx} points from x0={self.x0:g} to x1={self.x1:g} with dx={self.dx:g}"
        )
        out.append(
            f"y-axis: ny={self.ny} points from y0={self.y0:g} to y1={self.y1:g} with dy={self.dy:g}"
        )
        out.append(f"Number of grid points: {self.n}")
        return str.join("\n", out)

    @property
    def _one_line_str(self) -> str:
        return f"Grid2D (ny={self.ny}, nx={self.nx})"
