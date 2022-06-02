# Mesh


## Mesh functionality

The Mesh class is returned by `mikeio.open("my.mesh")` if the argument is a mesh file (or previously using `mikeio.Mesh()` given a mesh or a dfsu file). 

```python
>>> msh = mikeio.open("../tests/testdata/odense_rough.mesh")
>>> msh
Number of elements: 654
Number of nodes: 399
Projection: UTM-33
```

Apart from the common flexible file functionality, 
the Mesh object has the following methods and properties:

```{eval-rst}
.. autosummary::
    :nosignatures:

    mikeio.Mesh.write
    mikeio.Mesh.plot_boundary_nodes
    mikeio.Mesh.zn
```

See the [Mesh API specification](mikeio.Mesh) below for a detailed description. 



## Mesh example notebooks

See the [Mesh Example notebook](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Mesh.ipynb) for more Mesh operations (including shapely examples).



## Mesh API

```{eval-rst}
.. autoclass:: mikeio.Mesh
	:members:
	:inherited-members:
```

## Flexible Mesh Geometry API

See [Flexible Mesh Geometry API](GeometryFM)