.. _dfsu:

Dfsu and Mesh
*************

Dfsu and mesh files are both flexible mesh file formats used by MIKE 21/3 engines. 
The .mesh file is an ASCII format for storing the flexible mesh geometry. 
The .dfsu file is a binary dfs file with data on this mesh. The mesh geometry is 
available in a .dfsu file as static items.  

For a detailed description of the .mesh and .dfsu file specification see the `flexible file format documentation <https://manuals.mikepoweredbydhi.help/2021/General/FM_FileSpecification.pdf>`_.


The unstructered mesh
---------------------

The mesh geometry in a .mesh or a .dfsu file consists of a number of nodes and a number of elements.

Each node has:

* Node id
* X,Y,Z coordinate
* Code for the boundary

Each element has:

* Element id
* Element type; triangular, quadrilateral, prism etc.
* Element table; specifies for each element the nodes that defines the element. 


// The element table references a node by specifying the number in the list of nodes (not the index!). 2D elements specify their node in counter clockwise order.



Common Dfsu and Mesh properties
-------------------------------

MIKE IO has a `Dfsu class <#mikeio.Dfsu>`_ for handling .dfsu files 
and a `Mesh class <#mikeio.Mesh>`_ for handling .mesh files both they inherit from the 
same base class and have the same core functionality. 

* n_nodes - number of nodes
* node_coordinates - coordinates (x,y,z) of all nodes
* codes - node codes of all nodes (0=water, 1=land, ...)
* valid_codes - unique list of node codes
* boundary_codes - provides a unique list of boundary codes
* node_ids - a list of node ids (often trivial)
* boundary_polylines - Lists of closed polylines defining domain outline

* n_elements - number of elements
* element_ids - a list of element ids (often trivial)
* element_coordinates - center coordinates of each element
* element_table - element to node connectivity
* max_nodes_per_element - the max number of nodes for en element in this geometry

* projection_string - the projection string (e.g. 'LONG/LAT' or 'UTM32')
* is_geo - Are coordinates geographical (LONG/LAT)?
* is_local_coordinates - Are coordinates local (relative)


Main common Dfsu and Mesh methods
---------------------------------

* `contains() <#mikeio.Dfsu.contains>`_ - test if a list of points are contained by mesh
* `find_nearest_elements() <#mikeio.Dfsu.find_nearest_elements>`_ - find index of nearest elements (optionally for a list)
* `plot() <#mikeio.Dfsu.plot>`_ - plot unstructured data and/or mesh, mesh outline
* `to_shapely() <#mikeio.Dfsu.to_shapely>`_ - export flexible file geometry as shapely MultiPolygon
* get_overset_grid() - get a 2d grid that covers the domain by specifying spacing or shape
* get_2d_interpolant() - IDW interpolant for list of coordinates
* interp2d() - interp spatially in data (2d only)
* get_element_area() - Calculate the horizontal area of each element.
* elements_to_geometry() - export elements to new flexible file geometry

Mesh functionality
------------------

The Mesh class is initialized with a mesh or a dfsu file. 


.. code-block:: python

    >>> msh = Mesh("../tests/testdata/odense_rough.mesh")
    >>> msh
    Number of elements: 654
    Number of nodes: 399
    Projection: UTM-33


A part from the common flexible file functionality, 
the Mesh object has the following methods:

* `set_z() <#mikeio.Mesh.set_z>`_ - change the depth by setting the z value of each node
* `set_codes() <#mikeio.Mesh.set_codes>`_ - change the code values of the nodes
* `write() <#mikeio.Mesh.write>`_  - write mesh to file (will overwrite if file exists)
* `plot_boundary_nodes() <#mikeio.Mesh.plot_boundary_nodes>`_ - plot mesh boundary nodes and their codes

See the `Mesh API specification <#mikeio.Mesh>`_ below for a detailed description.


Dfsu functionality
------------------

The Dfsu class is initialized with a mesh or a dfsu file. 

A part from the common flexible file functionality, the Dfsu has the following properties:

* deletevalue - File delete value (NaN value)
* n_items - number of items
* items - List of items
* n_timesteps - Number of timesteps
* start_time - First time instance (as datetime)
* end_time - Last time instance (as datetime)
* is_equidistant - Is the time series equidistant in time
* timestep - Time step in seconds (if is_equidistant)
* is_tri_only - Does the mesh consist of triangles only?
* is_2d - type is either Dfsu2D (2 horizontal dimensions)


A part from the common flexible file functionality, the Dfsu has the following methods:

* read()
* write() - write a new dfsu file
* write_header() - write the header of a new dfsu file
* close() - finalize write for a dfsu file opened with write(…,keep_open=True)
* elements_to_geometry() 
* extract_track() - extract track data from a dfsu file


Dfsu types
----------


2D horizontal dfsu files
------------------------



Layered dfsu files
------------------ 

3D dfsu and 2d vertical slices 


Additional properties for layered files: 

* n_layers - maximum number of layers
* n_sigma_layers - number of sigma layers
* n_z_layers - maximum number of z-layers
* layer_ids - the layer number for each 3d element
* top_elements - list of 3d element ids of surface layer
* bottom_elements - list of 3d element ids of bottom layer
* n_layers_per_column - list of number of layers for each column
* geometry2d - the 2d geometry for a 3d object
* e2_e3_table - the 2d-to-3d element connectivity table for a 3d object
* elem2d_ids - the associated 2d element id for each 3d element

Additional methods for layered files: 

* get_layer_elements() - 3d element ids for one (or more) specific layer(s)
* find_nearest_profile_elements(x, y) - find 3d elements of profile nearest to (x,y) coordinates

plot_vertical_profile()


Dfsu API
--------
.. autoclass:: mikeio.Dfsu
	:members:
	:inherited-members:

Mesh API
--------
.. autoclass:: mikeio.Mesh
	:members:
	:inherited-members: