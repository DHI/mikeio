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

In addition to the common [dfsu-geometry properties and methods](./dfu-mesh-overview.md#mike-io-flexible-mesh-geometry), `Mesh` has the following properties and methods:


```{eval-rst}
.. autosummary::
    :nosignatures:

    mikeio.Mesh.write
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

A mesh object 

```{eval-rst}
.. autoclass:: mikeio.spatial.FM_geometry.GeometryFM
	:members:
	:inherited-members:
```

## FM Geometry Plotter API

```{eval-rst}
.. autoclass:: mikeio.spatial.FM_geometry._GeometryFMPlotter
	:members:
	:inherited-members:
```