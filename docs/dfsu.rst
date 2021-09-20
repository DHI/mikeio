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



Common Dfsu and Mesh properties
-------------------------------

MIKE IO has a `Dfsu class <#mikeio.Dfsu>`_ for handling .dfsu files 
and a `Mesh class <#mikeio.Mesh>`_ for handling .mesh files both they inherit from the 
same base class and have the same core functionality. 

* n_elements
* n_nodes


Main common Dfsu and Mesh methods
---------------------------------

* contains()
* find_nearest_elements()
* plot()


Dfsu types
----------

x

Layered dfsu files
------------------ 

x



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