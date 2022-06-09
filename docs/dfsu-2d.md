# Dfsu 2D


## Dfsu functionality

A Dfsu class (e.g. Dfsu2DH) is returned by {py:meth}`mikeio.open` if the argument is a dfsu file. 

Apart from the common [dfsu-geometry properties and methods](./dfu-mesh-overview.md#mike-io-flexible-mesh-geometry), Dfsu2DH has the following *properties*:


```{eval-rst}
.. autosummary::
    :nosignatures:

    mikeio.dfsu._Dfsu.deletevalue
    mikeio.dfsu._Dfsu.n_items
    mikeio.dfsu._Dfsu.items
    mikeio.dfsu._Dfsu.n_timesteps
    mikeio.dfsu._Dfsu.start_time
    mikeio.dfsu._Dfsu.end_time
    mikeio.dfsu._Dfsu.timestep
    mikeio.dfsu._Dfsu.is_2d
```

Dfsu2DH has the following *methods*:

```{eval-rst}
.. autosummary::
    :nosignatures:

    mikeio.dfsu._Dfsu.read
    mikeio.dfsu._Dfsu.write
    mikeio.dfsu._Dfsu.write_header
    mikeio.dfsu._Dfsu.close
```

See the [API specification](Dfsu2DH) below for a detailed description. 

See the [Dfsu Read Example notebook](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfsu%20-%20Read.ipynb) for basic dfsu functionality.



## Dfsu 2DH API

```{eval-rst}
.. autoclass:: mikeio.dfsu.Dfsu2DH
	:members:
	:inherited-members:
```


## Flexible Mesh Geometry API

See [Flexible Mesh Geometry API](GeometryFM)


## FM Geometry Plotter API

See [FM Geometry Plotter API](_GeometryFMPlotter)

## DataArray Plotter FM API

A DataArray `da` with a GeometryFM geometry can be plotted using `da.plot`. 

```{eval-rst}
.. autoclass:: mikeio.dataarray._DataArrayPlotterFM
	:members:
	:inherited-members:
```