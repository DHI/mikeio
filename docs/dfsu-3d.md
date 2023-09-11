# Dfsu 3D


In addition to the common [dfsu-geometry properties and methods](MIKE IO Flexible Mesh Geometry), Dfsu3D has the below additional *properties* (from it's geometry [GeometryFM3D](FM Geometry 3D API): 

```{eval-rst}
.. autosummary::
    :nosignatures:

    mikeio.dfsu._layered.Dfsu3D.n_layers
    mikeio.dfsu._layered.Dfsu3D.n_sigma_layers
    mikeio.dfsu._layered.Dfsu3D.n_z_layers
    mikeio.dfsu._layered.Dfsu3D.layer_ids
    mikeio.dfsu._layered.Dfsu3D.top_elements
    mikeio.dfsu._layered.Dfsu3D.bottom_elements
    mikeio.dfsu._layered.Dfsu3D.n_layers_per_column
    mikeio.dfsu._layered.Dfsu3D.geometry2d
    mikeio.dfsu._layered.Dfsu3D.e2_e3_table
    mikeio.dfsu._layered.Dfsu3D.elem2d_ids
```


And in addition to from the basic dfsu functionality, Dfsu3D has the below additional *methods*: 

```{eval-rst}
.. autosummary::
    :nosignatures:

    mikeio.dfsu._layered.Dfsu3D.get_layer_elements
```



```{warning}
In MIKE Zero, layer ids are 1-based. In MIKE IO, all ids are **0-based**following standard Python indexing. The bottom layer is 0. In early versionsof MIKE IO, layer ids was 1-based! From release 0.10 all ids are 0-based.
```


## Dfsu 3D example notebooks

See the [Dfsu - 3D sigma-z.ipynb](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfsu%20-%203D%20sigma-z.ipynb) for 3d dfsu functionality.


## Dfsu 3D API

```{eval-rst}
.. autoclass:: mikeio.dfsu._layered.Dfsu3D
	:members:
	:inherited-members:
```

## FM Geometry 3D API

```{eval-rst}
.. autoclass:: mikeio.spatial._FM_geometry_layered.GeometryFM3D
	:members:
	:inherited-members:
```

