import numpy as np
from DHI.Generic.MikeZero.DFS.mesh import MeshFile, MeshBuilder
from DHI.Generic.MikeZero import eumQuantity
from .eum import ItemInfo, EUMType, EUMUnit
from .dotnet import asnetarray_v2

def min_horizontal_dist_meters(coords, targets, is_geo=False):
    xe = coords[:,0]
    ye = coords[:,1]
    n = len(xe)
    d  = np.zeros(n)
    for j in range(n):
        d1 = dist_in_meters(targets, [xe[j], ye[j]], is_geo=is_geo)
        d[j] = d1.min()
    return d

def dist_in_meters(coords, pt, is_geo=False):
    xe = coords[:,0]
    ye = coords[:,1]
    xp = pt[0]
    yp = pt[1]    
    if is_geo:        
        d = _get_dist_geo(xe, ye, xp, yp)
    else:   
        d = np.sqrt(np.square(xe - xp) + np.square(ye - yp))
    return d

def _get_dist_geo(lon, lat, lon1, lat1):
    # assuming input in degrees!
    R = 6371e3 # Earth radius in metres
    dlon = np.deg2rad(lon1 - lon)
    if(np.any(dlon>np.pi)): 
        dlon = dlon - 2*np.pi
    if(np.any(dlon<-np.pi)): 
        dlon = dlon + 2*np.pi    
    dlat = np.deg2rad(lat1 - lat)
    x = dlon*np.cos(np.deg2rad((lat+lat1)/2))
    y = dlat
    d = R * np.sqrt(np.square(x) + np.square(y) )
    return d

class Grid2D:
    """2D grid
    """
    _x = None
    _x0 = None
    _x1 = None
    _dx = None
    _nx = None
    _y = None
    _y0 = None
    _y1 = None
    _dy = None
    _ny = None

    _xx = None
    _yy = None
    
    @property
    def x(self):
        """array of x-coordinates (single row)
        """
        return self._x

    @property
    def x0(self):
        """left end-point
        """
        return self._x0

    @property
    def x1(self):
        """right end-point
        """        
        return self._x1

    @property
    def dx(self):
        """x-spacing
        """
        return self._dx 

    @property
    def nx(self):
        """number of points in x-direction
        """        
        return self._nx
    
    @property
    def y(self):
        """array of y-coordinates (single column)
        """        
        return self._y

    @property
    def y0(self):
        """lower end-point
        """
        return self._y0

    @property
    def y1(self):
        """upper end-point
        """
        return self._y1

    @property
    def dy(self):
        """y-spacing
        """        
        return self._dy 

    @property
    def ny(self):
        """number of points in y-direction
        """        
        return self._ny

    @property
    def n(self):
        """total number of grid points
        """        
        return self._nx*self._ny

    @property
    def xx(self):
        """2d array of all x-coordinates 
        """
        if self._xx is None:
            self._create_meshgrid(self.x, self.y)
        return self._xx

    @property
    def yy(self):
        """2d array of all y-coordinates 
        """
        if self._yy is None:
            self._create_meshgrid(self.x, self.y)
        return self._yy

    @property
    def xy(self):
        """ n-by-2 array of x- and y-coordinates 
        """
        xcol = self.xx.reshape(-1,1)
        ycol = self.yy.reshape(-1,1)
        return np.column_stack([xcol, ycol])

    @property
    def coordinates(self):
        """ n-by-2 array of x- and y-coordinates 
        """
        return self.xy

    @property
    def bbox(self):   
        """bounding box [x0, y0, x1, y1]
        """             
        return [self._x0, self._y0, self._x1, self._y1]
 
    def __init__(self, x=None, y=None, bbox=None, dxdy=None, shape=None):
        """create 2d grid 

        Parameters
        ----------
        x : array-like, optional
            1d array of x-coordinates
        y : array-like, optional
            1d array of y-coordinates
        bbox : array(float), optional
            [x0, y0, x1, y1]
        dxdy : float or (float, float), optional
            grid resolution in x- and y-direction
        shape : (int, int), optional
            tuple with nx and ny describing number of points in each direction
            one of them can be None, in which case the value will be inferred

        Examples
        --------
        >>> g = Grid2D(bbox=[0,0,10,20], dxdy=0.25)

        >>> g = Grid2D(bbox=[0,0,10,20], shape=(5,10))
        
        >>> x = np.linspace(0.0, 1000.0, 201)
        >>> y = [0, 2.0]
        >>> g = Grid2D(x, y)
        
        """
        if (x is not None) and (len(x)==4):
            # first positional argument 'x' is probably bbox
            if (y is None) or (len(y)!=4):
                bbox, x = x, bbox

        if bbox is not None:
            self._create_in_bbox(bbox, dxdy, shape)
        elif (x is not None) and (y is not None):
            self._create_from_x_and_y(x, y)
        else:
            raise ValueError('Please provide either bbox or both x and y')

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
        x0 = bbox[0]
        y0 = bbox[1]
        x1 = bbox[2]
        y1 = bbox[3]
        xr = x1 - x0
        yr = y1 - y0
        
        if (dxdy is None) and (shape is None):
            if xr <= yr:
                nx = 10
                ny = int(np.ceil(nx*yr/xr))
            else:
                ny = 10
                nx = int(np.ceil(ny*xr/yr))
            dx = xr/(nx-1)
            dy = yr/(ny-1)
        else:
            if shape is not None:
                if len(shape) != 2:
                    raise ValueError('shape must be (nx,ny)')
                nx, ny = shape
                if (nx is None) and (ny is None):
                    raise ValueError('nx and ny cannot both be None')
                if nx is None:
                    nx = int(np.ceil(ny*xr/yr))
                if ny is None:
                    ny = int(np.ceil(nx*yr/xr))                
                dx = xr/(nx-1)
                dy = yr/(ny-1)
            elif dxdy is not None:
                if np.isscalar(dxdy):
                    dy = dx = dxdy
                else:
                    dx, dy = dxdy
                nx = int(np.ceil(xr/dx)) + 1
                ny = int(np.ceil(yr/dy)) + 1
            else:
                raise ValueError('dxdy and shape cannot both be provided! Chose one.')
        
        self._x0 = x0        
        self._dx = dx
        self._nx = nx
        self._create_x_axis(x0, dx, nx)
        
        self._y0 = y0
        self._dy = dy
        self._ny = ny
        self._create_y_axis(y0, dy, ny)

    def _create_from_x_and_y(self, x, y):
        
        self._x0 = x[0]
        self._x1 = x[-1]
        self._nx = len(x)
        self._dx = x[1]-x[0]
        self._x = x
    
        self._y0 = y[0]
        self._y1 = y[-1]
        self._ny = len(y)
        self._dy = y[1]-y[0]
        self._y = y
        self._xx, self._yy = None, None

    def _create_x_axis(self, x0, dx, nx):
        self._x1 = x0 + dx*(nx-1)
        self._x = np.linspace(x0, self._x1, nx)
        self._xx, self._yy = None, None

    def _create_y_axis(self, y0, dy, ny):
        self._y1 = y0 + dy*(ny-1)
        self._y = np.linspace(y0, self._y1, ny)
        self._xx, self._yy = None, None

    def _create_meshgrid(self, x, y):
        self._xx, self._yy = np.meshgrid(x, y)

    def contains(self, xy):
        """test if a list of points are inside grid

        Parameters
        ----------
        points : array-like n-by-2
            x,y-coordinates of n points to be tested

        Returns
        -------
        bool array
            True for points inside, False otherwise
        """
        xp = xy[:,0]
        yp = xy[:,1]
        xinside = (self.x0 <= xp) & (xp <= self.x1)
        yinside = (self.y0 <= yp) & (yp <= self.y1)
        return xinside & yinside

    def _to_element_table(self, index_base=0):

        elem_table = []
        for elx in range(self.nx-1):
            # each col
            for ely in range(self.ny-1):
                # each row (in this col)
                n1 = ely*self.nx + elx  + index_base
                n2 = (ely+1)*self.nx + elx  + index_base
                elem_table.append([n1, n1+1, n2+1, n2])
        return elem_table

    def to_mesh(self, outfilename, projection=None, z=None):
        if projection is None:
            projection = "LONG/LAT"

        x  = self.xy[:,0]
        y  = self.xy[:,1]
        if z is None:
            z  = np.zeros(self.n)
        codes = np.zeros(self.n, dtype=int)
        codes[y==self.y1] = 5   # north   
        codes[x==self.x1] = 4   # east
        codes[y==self.y0] = 3   # south
        codes[x==self.x0] = 2   # west
        codes[(y==self.y1) & (x==self.x0)] = 5   # corner->north

        builder = MeshBuilder()        
        builder.SetNodes(x, y, z, codes)

        elem_table = self._to_element_table(index_base=1)
        builder.SetElements(asnetarray_v2(elem_table))

        builder.SetProjection(projection)
        quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)
        builder.SetEumQuantity(quantity)
        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)

    @staticmethod
    def xy_to_bbox(xy, buffer=None):
        """return bounding box for list of coordinates
        """
        if buffer is None:
            buffer = 0       
        x0 = xy[:,0].min() + buffer
        y0 = xy[:,1].min() + buffer
        x1 = xy[:,0].max() - buffer
        y1 = xy[:,1].max() - buffer
        return [x0, y0, x1, y1]

    def __repr__(self):
        out = []
        out.append("2D Grid")
        out.append(f"x-axis: nx={self.nx} points from x0={self.x0:g} to x1={self.x1:g} with dx={self.dx:g}")
        out.append(f"y-axis: ny={self.ny} points from y0={self.y0:g} to y1={self.y1:g} with dy={self.dy:g}")
        out.append(f"Number of grid points: {self.n}")        
        return str.join("\n", out)

