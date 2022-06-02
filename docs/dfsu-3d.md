# Dfsu 3D



## Layered dfsu files

There are three type of layered dfsu files: 3D dfsu, 2d vertical slices and 1d vertical profiles.

Apart from the basic dfsu functionality, layered dfsu have the below additional *properties* (from the geometry): 

```{eval-rst}
.. autosummary::
    :nosignatures:

    mikeio.dfsu_layered.DfsuLayered.n_layers
    mikeio.dfsu_layered.DfsuLayered.n_sigma_layers
    mikeio.dfsu_layered.DfsuLayered.n_z_layers
    mikeio.dfsu_layered.DfsuLayered.layer_ids
    mikeio.dfsu_layered.DfsuLayered.top_elements
    mikeio.dfsu_layered.DfsuLayered.bottom_elements
    mikeio.dfsu_layered.DfsuLayered.n_layers_per_column
    mikeio.dfsu_layered.Dfsu3D.geometry2d
    mikeio.dfsu_layered.DfsuLayered.e2_e3_table
    mikeio.dfsu_layered.DfsuLayered.elem2d_ids
```

Dfsu3D addtionally, has:

```{eval-rst}
.. autosummary::
    :nosignatures:

    mikeio.dfsu_layered.Dfsu3D.geometry2d
```



Apart from the basic dfsu functionality, Dfsu3D has the below additional *methods*: 

```{eval-rst}
.. autosummary::
    :nosignatures:

    mikeio.dfsu_layered.DfsuLayered.get_layer_elements
    mikeio.dfsu_layered.Dfsu3D.find_nearest_profile_elements
```



```{warning}
In MIKE Zero, layer ids are 1-based. In MIKE IO, all ids are **0-based**following standard Python indexing. The bottom layer is 0. In early versionsof MIKE IO, layer ids was 1-based! From release 0.10 all ids are 0-based.
```


## Dfsu 3D example notebooks


See the [Dfsu - 3D sigma-z.ipynb](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfsu%20-%203D%20sigma-z.ipynb) for 3d dfsu functionality.


## Dfsu 3D API

```{eval-rst}
.. autoclass:: mikeio.dfsu_layered.Dfsu3D
	:members:
	:inherited-members:
```

## FM Geometry 3D API

```{eval-rst}
.. autoclass:: mikeio.spatial.FM_geometry.GeometryFM3D
	:members:
	:inherited-members:
```
