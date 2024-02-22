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

See the [Mesh API specification](`mikeio.Mesh`) for a detailed description. 


## Mesh example notebooks

See the [Mesh Example notebook](examples/Mesh.qmd) for more Mesh operations (including shapely examples).


