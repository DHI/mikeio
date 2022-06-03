# Dfsu Spectral


MIKE 21 SW can output spectral information in *points*, along *lines* or in an *area*. If the full (2d) spectra are stored, the dfsu files will have two additional axes: frequency and directions. 


## Spectral Dfsu example notebooks

* [Dfsu - Spectral data.ipynb](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfsu%20-%20Spectral%20data.ipynb) 
* [Dfsu - Spectral data other formats.ipynb](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfsu%20-%20Spectral%20data%20other%20formats.ipynb) 



## Dfsu Spectral API

```{eval-rst}
.. autoclass:: mikeio.dfsu_spectral.DfsuSpectral
	:members:
	:inherited-members:
```


## FM Geometry Point Spectrum API

```{eval-rst}
.. autoclass:: mikeio.spatial.FM_geometry.GeometryFMPointSpectrum
	:members:
	:inherited-members:
```

## FM Geometry Line Spectrum API

```{eval-rst}
.. autoclass:: mikeio.spatial.FM_geometry.GeometryFMLineSpectrum
	:members:
	:inherited-members:
```

## FM Geometry Area Spectrum API

```{eval-rst}
.. autoclass:: mikeio.spatial.FM_geometry.GeometryFMAreaSpectrum
	:members:
	:inherited-members:
```

