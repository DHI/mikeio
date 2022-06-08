

# Dfs3

A dfs3 file contains 3D gridded data.  


```python
>>> import mikeio
>>> ds = mikeio.read("dissolved_oxygen.dfs3")
>>> ds
<mikeio.Dataset>
dims: (time:1, z:17, y:112, x:91)
time: 2001-12-28 00:00:00 (time-invariant)
geometry: Grid3D(nz=17, ny=112, nx=91)
items:
  0:  Diss. oxygen (mg/l) <Concentration 3> (mg per liter)
```

A specific layer can be read with the `layers` argument, in which case a 2D Dataset will be returned: 

```python
>>> import mikeio
>>> mikeio.read("dissolved_oxygen.dfs2", layers="bottom")
<mikeio.Dataset>
dims: (time:1, y:112, x:91)
time: 2001-12-28 00:00:00 (time-invariant)
geometry: Grid2D (ny=112, nx=91)
items:
  0:  Diss. oxygen (mg/l) <Concentration 3> (mg per liter)
```

## Grid3D

The spatial information is available in the `geometry` attribute (accessible from Dfs3, Dataset, and DataArray), which in the case of a dfs3 file is a [`Grid3D`](Grid3D) geometry. 

```python
>>> dfs = mikeio.open("dissolved_oxygen.dfs3")
>>> dfs.geometry
<mikeio.Grid3D>
x: [0, 150, ..., 1.35e+04] (nx=91, dx=150)
y: [0, 150, ..., 1.665e+04] (ny=112, dy=150)
z: [0, 1, ..., 16] (nz=17, dz=1)
origin: (10.37, 55.42), orientation: 18.125
projection: PROJCS["UTM-32",GEOGCS["Unused",DATUM["UTM...
```

Grid3D's primary properties and methods are: 

* `x` 
* `nx`
* `dx`
* `y`
* `ny`
* `dy`
* `z`
* `nz`
* `dz`
* `origin`
* `projection`
* `contains()`
* `isel()`

See [API specification](Grid3D) below for details.

## Dfs3 Example notebooks

* [Dfs3-Basic](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs3%20-%20Basic.ipynb)  



## Dfs3 API

```{eval-rst}
.. autoclass:: mikeio.Dfs3
	:members:
	:inherited-members:
```

## Grid3D API

```{eval-rst}
.. autoclass:: mikeio.Grid3D
	:members:
	:inherited-members:
```

