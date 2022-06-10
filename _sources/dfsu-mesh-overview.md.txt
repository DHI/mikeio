# Dfsu and Mesh Overview

Dfsu and mesh files are both flexible mesh file formats used by MIKE 21/3 engines. 
The .mesh file is an ASCII file for storing the flexible mesh geometry. 
The .dfsu file is a binary dfs file with data on this mesh. The mesh geometry is 
available in a .dfsu file as static items.  

For a detailed description of the .mesh and .dfsu file specification see the [flexible file format documentation](https://manuals.mikepoweredbydhi.help/2021/General/FM_FileSpecification.pdf).


## The flexible mesh

The mesh geometry in a .mesh or a .dfsu file consists of a list of nodes and a list of elements.

Each node has:

* Node id
* x,y,z coordinates
* Code (0 for internal water points, 1 for land, >1 for open boundary)

Each element has:

* Element id
* Element table; specifies for each element the nodes that defines the element. 
(the number of nodes defines the type: triangular, quadrilateral, prism etc.)


```{warning} 
In MIKE Zero, node ids, element ids and layer ids are 1-based.  In MIKE IO, all ids are **0-based** following standard Python indexing. That means, as an example, that when finding the element closest to a point its id will be 1 lower in MIKE IO compared to examining the file in MIKE Zero.
```

## MIKE IO Flexible Mesh Geometry 

MIKE IO has a Flexible Mesh Geometry class, [`GeometryFM`](GeometryFM), containing the list of node coordinates and the element table which defines the mesh, as well as a number of derived properties (e.g. element coordinates) and methods making it convenient to work with the mesh. 

| Property  |      Description     |
|----------|--------------|
| `n_nodes` | Number of nodes | 
| `node_coordinates` | Coordinates (x,y,z) of all nodes | 
| `codes` | Codes of all nodes (0:water, 1:land, >=2:open boundary) | 
| `boundary_polylines` | Lists of closed polylines defining domain outline | 
| `n_elements` | Number of elements | 
| `element_coordinates` | Center coordinates of each element | 
| `element_table` | Element to node connectivity | 
| `max_nodes_per_element` | The maximum number of nodes for an element | 
| `is_tri_only` | Does the mesh consist of triangles only? | 
| `projection_string` | The projection string | 
| `is_geo` | Are coordinates geographical (LONG/LAT)? | 
| `is_local_coordinates` | Are coordinates relative (NON-UTM)? | 
| `type_name` | Type name, e.g. Dfsu2D| 


| Method  |      Description     |
|----------|--------------|
| `contains()` | test if a list of points are contained by mesh | 
| `find_index()` | Find index of elements containing points/area|
| `isel()` | Get subset geometry for list of indicies |
| `find_nearest_points()` | Find index of nearest elements (optionally for a list) |
| `plot` | Plot the geometry |
| `get_overset_grid()` | Get a Grid2D covering the domain |
| `to_shapely()` | Export mesh as shapely MultiPolygon | 
| `get_element_area()` | Calculate the horizontal area of each element | 


These properties and methods are accessible from the geometry, but also from the Mesh/Dfsu object. 

If a .dfsu file is *read* with {py:meth}`mikeio.read`, the returned Dataset ds will contain a Flexible Mesh Geometry `geometry`. If a .dfsu or a .mesh file is *opened* with {py:meth}`mikeio.open`, the returned object will also contain a Flexible Mesh Geometry `geometry`. 

```python
>>> import mikeio
>>> ds = mikeio.read("oresundHD_run1.dfsu")
>>> ds.geometry
Flexible Mesh Geometry: Dfsu2D
number of nodes: 2046
number of elements: 3612
projection: UTM-33

>>> dfs = mikeio.open("oresundHD_run1.dfsu")
>>> dfs.geometry
Flexible Mesh Geometry: Dfsu2D
number of nodes: 2046
number of elements: 3612
projection: UTM-33
```




## Common Dfsu and Mesh properties

MIKE IO has Dfsu classes for .dfsu files 
and a [Mesh class](mikeio.Mesh) for .mesh files which both 
have a [Flexible Mesh Geometry](GeometryFM) accessible through the ´geometry´ accessor. 




## Dfsu types

The following dfsu file types are supported by MIKE IO.

* 2D horizontal. 
* 3D layered. 
* 2D vertical profile - a vertical slice through a 3D layered file. 
* 1D vertical column - a vertical dfs1 file and is produced by taking out one column of a 3D layered file.
* 3D/4D SW, two horizontal dimensions and 1-2 spectral dimensions. Output from MIKE 21 SW.

When a dfsu file is opened with mikeio.open() the returned dfs object will be a specialized class [Dfsu2DH](Dfsu2DH), [Dfsu3D](Dfsu3D), [Dfsu2DV](Dfsu2DV), or [DfsuSpectral](DfsuSpectral) according to the type of dfsu file. 

The layered files (3d, 2d/1d vertical) can have both sigma- and z-layers or only sigma-layers. 

In most cases values are stored in cell centers and vertical (z) information in nodes, 
but the following values types exists: 

* Standard value type, storing values on elements and/or nodes. This is the default type.
* Face value type, storing values on element faces. This is used e.g. for HD decoupling files, to store the discharge between elements.
* Spectral value type, for each node or element, storing vales for a number of frequencies and/or directions. This is the file type for spectral output from the MIKE 21 SW. 


