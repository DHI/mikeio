.. _dfsu:

Dfsu and Mesh
*************

.. warning::
    TODO Not updated to MIKE IO 1.0

Dfsu and mesh files are both flexible mesh file formats used by MIKE 21/3 engines. 
The .mesh file is an ASCII format for storing the flexible mesh geometry. 
The .dfsu file is a binary dfs file with data on this mesh. The mesh geometry is 
available in a .dfsu file as static items.  

For a detailed description of the .mesh and .dfsu file specification see the `flexible file format documentation <https://manuals.mikepoweredbydhi.help/2021/General/FM_FileSpecification.pdf>`_.


The flexible mesh
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

.. warning::
    In MIKE Zero, node ids, element ids and layer ids are 1-based. 
    In MIKE IO, all ids are **0-based** following standard Python indexing. 
    That means, as an example, that when finding the element closest to a 
    point its id will be 1 lower in MIKE IO compared to examining the file in MIKE Zero.



Common Dfsu and Mesh properties
-------------------------------

MIKE IO has a `Dfsu class <#mikeio.Dfsu>`_ for handling .dfsu files 
and a `Mesh class <#mikeio.Mesh>`_ for handling .mesh files both they inherit from the 
same base class and have the same core functionality. 

.. autosummary::
    :nosignatures:

    mikeio.Mesh.n_nodes
    mikeio.Mesh.node_coordinates
    mikeio.Mesh.codes
    mikeio.Mesh.boundary_polylines
    mikeio.Mesh.n_elements
    mikeio.Mesh.element_coordinates
    mikeio.Mesh.element_table
    mikeio.Mesh.max_nodes_per_element
    mikeio.Mesh.is_tri_only
    mikeio.Mesh.projection_string
    mikeio.Mesh.is_geo
    mikeio.Mesh.is_local_coordinates
    mikeio.Mesh.type_name    


Common Dfsu and Mesh methods
----------------------------


.. autosummary::
    :nosignatures:

    mikeio.Mesh.contains
    mikeio.Mesh.find_nearest_elements
    mikeio.Mesh.plot
    mikeio.Mesh.to_shapely
    mikeio.Mesh.get_overset_grid
    mikeio.Mesh.get_2d_interpolant
    mikeio.Mesh.interp2d
    mikeio.Mesh.get_element_area
    mikeio.Mesh.elements_to_geometry


Mesh functionality
------------------

The Mesh class is initialized with a mesh or a dfsu file. 


.. code-block:: python

    >>> msh = Mesh("../tests/testdata/odense_rough.mesh")
    >>> msh
    Number of elements: 654
    Number of nodes: 399
    Projection: UTM-33


Apart from the common flexible file functionality, 
the Mesh object has the following methods and properties:

.. autosummary::
    :nosignatures:

    mikeio.Mesh.write
    mikeio.Mesh.plot_boundary_nodes
    mikeio.Mesh.zn

See the `Mesh API specification <#mikeio.Mesh>`_ below for a detailed description. 
See the `Mesh Example notebook <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Mesh.ipynb>`_ for more Mesh operations (including shapely examples).


Dfsu functionality
------------------

The Dfsu class is initialized with a mesh or a dfsu file. 

Apart from the common flexible file functionality, the Dfsu has the following *properties*:

.. autosummary::
    :nosignatures:

    mikeio.Dfsu.deletevalue
    mikeio.Dfsu.n_items
    mikeio.Dfsu.items
    mikeio.Dfsu.n_timesteps
    mikeio.Dfsu.start_time
    mikeio.Dfsu.end_time
    mikeio.Dfsu.timestep
    mikeio.Dfsu.is_2d


Apart from the common flexible file functionality, the Dfsu has the following *methods*:

.. autosummary::
    :nosignatures:

    mikeio.Dfsu.read
    mikeio.Dfsu.write
    mikeio.Dfsu.write_header
    mikeio.Dfsu.close
    mikeio.Dfsu.extract_track

See the `Dfsu API specification <#mikeio.Dfsu>`_ below for a detailed description. 
See the `Dfsu Read Example notebook <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfsu%20-%20Read.ipynb>`_ for basic dfsu functionality.



Dfsu types
----------

The following dfsu file types are supported by MIKE IO.

* 2D horizontal. 
* 3D layered. 
* 2D vertical profile - a vertical slice through a 3D layered file. 
* 1D vertical column - a vertical dfs1 file and is produced by taking out one column of a 3D layered file.
* 3D/4D SW, two horizontal dimensions and 1-2 spectral dimensions. Output from MIKE 21 SW.

The layered files (3d, 2d/1d vertical) can have both sigma- and z-layers or only sigma-layers. 

In most cases values are stored in cell centers and vertical (z) information in nodes, 
but the following values types exists: 

* Standard value type, storing values on elements and/or nodes. This is the default type.
* Face value type, storing values on element faces. This is used e.g. for HD decoupling files, to store the discharge between elements.
* Spectral value type, for each node or element, storing vales for a number of frequencies and/or directions. This is the file type for spectral output from the MIKE 21 SW. 




Layered dfsu files
------------------ 

There are three type of layered dfsu files: 3D dfsu, 2d vertical slices and 1d vertical profiles.

Apart from the basic dfsu functionality, layered dfsu have the below additional *properties*: 

.. autosummary::
    :nosignatures:

    mikeio.Dfsu.n_layers
    mikeio.Dfsu.n_sigma_layers
    mikeio.Dfsu.n_z_layers
    mikeio.Dfsu.layer_ids
    mikeio.Dfsu.top_elements
    mikeio.Dfsu.bottom_elements
    mikeio.Dfsu.n_layers_per_column
    mikeio.Dfsu.geometry2d
    mikeio.Dfsu.e2_e3_table
    mikeio.Dfsu.elem2d_ids

Apart from the basic dfsu functionality, layered dfsu have the below additional *methods*: 

.. autosummary::
    :nosignatures:

    mikeio.Dfsu.get_layer_elements
    mikeio.Dfsu.find_nearest_profile_elements
    mikeio.Dfsu.plot_vertical_profile

.. warning::
    In MIKE Zero, layer ids are 1-based. 
    In MIKE IO, all ids are **0-based** following standard Python indexing. 
    The bottom layer is 0. 
    In previous versions of MIKE IO, layer ids was 1-based! 
    From release 0.10 all ids are 0-based.  



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