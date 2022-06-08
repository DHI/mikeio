# Dfsu 2DV Vertical Profile


In addition to the common [dfsu-geometry properties and methods](./dfu-mesh-overview.md#mike-io-flexible-mesh-geometry), Dfsu2DV has the below additional *properties* (from it's geometry [GeometryFMVerticalProfile](GeometryFMVerticalProfile)): 



```{eval-rst}
.. autosummary::
    :nosignatures:

    mikeio.dfsu_layered.Dfsu2DV.n_layers
    mikeio.dfsu_layered.Dfsu2DV.n_sigma_layers
    mikeio.dfsu_layered.Dfsu2DV.n_z_layers
    mikeio.dfsu_layered.Dfsu2DV.layer_ids
    mikeio.dfsu_layered.Dfsu2DV.top_elements
    mikeio.dfsu_layered.Dfsu2DV.bottom_elements
    mikeio.dfsu_layered.Dfsu2DV.n_layers_per_column
    mikeio.dfsu_layered.Dfsu2DV.e2_e3_table
    mikeio.dfsu_layered.Dfsu2DV.elem2d_ids
```


And in addition to the basic dfsu functionality, Dfsu2DV has the below additional *methods*: 

```{eval-rst}
.. autosummary::
    :nosignatures:

    mikeio.dfsu_layered.Dfsu2DV.get_layer_elements    
```



```{warning}
In MIKE Zero, layer ids are 1-based. In MIKE IO, all ids are **0-based**following standard Python indexing. The bottom layer is 0. In early versionsof MIKE IO, layer ids was 1-based! From release 0.10 all ids are 0-based.
```

## Vertical Profile Dfsu example notebooks

* [Dfsu - Vertical Profile.ipynb](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfsu%20-%20Vertical%20Profile.ipynb) 



## Dfsu 2DV Vertical Profile API

```{eval-rst}
.. autoclass:: mikeio.dfsu_layered.Dfsu2DV
	:members:
	:inherited-members:
```

## FM Geometry 2DV Vertical Profile API

```{eval-rst}
.. autoclass:: mikeio.spatial.FM_geometry.GeometryFMVerticalProfile
	:members:
	:inherited-members:
```

## DataArray Plotter FM Vertical Profile API

A DataArray `da` with a GeometryFMVerticalProfile geometry can be plotted using `da.plot`. 

```{eval-rst}
.. autoclass:: mikeio.dataarray._DataArrayPlotterFMVerticalProfile
	:members:
	:inherited-members:
```