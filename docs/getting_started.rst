.. _getting_started:

Getting started
###############

After installing MIKE IO 

Resources
*********

* Online book: `Getting started with Dfs files in Python using MIKE IO <https://dhi.github.io/getting-started-with-mikeio>`_
* `Example notebooks <https://nbviewer.jupyter.org/github/DHI/mikeio/tree/main/notebooks/>`_
* `Unit tests <https://github.com/DHI/mikeio/tree/main/tests>`_
* `DFS file system specification <https://docs.mikepoweredbydhi.com/core_libraries/dfs/dfs-file-system/>`_


Dataset
*******
The MIKE IO `Dataset <api.html#mikeio.Dataset>`_ is a common data structure for data read from dfs files. 
All `read()` methods in MIKE IO returns a Dataset with the three main properties:

* items - a list of `ItemInfo <api.html#mikeio.eum.ItemInfo>`_ with name, type and unit of each item
* time - a pandas.DateTimeIndex with the time instances of the data
* data - a list of NumPy arrays---one for each item

Read more on the `Dataset page <dataset.html>`_.


Dfs0
****
A dfs0 file is also called a time series file. The MIKE IO `Dfs0 class <api.html#mikeio.Dfs0>`_ provide functionality for working with dfs0 files.  

.. code-block:: python

   from mikeio import Dfs0
   


Dfs2
****
A dfs2 file is also called a grid series file. Values in a dfs2 file are ‘element based’, i.e. values are defined in the centre of each grid cell. 
The MIKE IO `Dfs2 class <api.html#mikeio.Dfs2>`_ provide functionality for working with dfs2 files.  

.. code-block:: python

   from mikeio import Dfs2
   

Generic dfs
***********
MIKE IO has `generic dfs <api.html#module-mikeio.generic>`_ functionality that works for all dfs files: 

* `read() <api.html#mikeio.generic.extract>`_ - Read all data to a Dataset
* `concat() <api.html#mikeio.generic.extract>`_ - Concatenates files along the time axis
* `extract() <api.html#mikeio.generic.extract>`_ - Extract timesteps and/or items to a new dfs file
* `diff() <api.html#mikeio.generic.diff>`_ - Calculate difference between two dfs files
* `sum() <api.html#mikeio.generic.extract>`_ - Calculate the sum of two dfs files
* `scale() <api.html#mikeio.generic.extract>`_ - Apply scaling to any dfs file

All methods except read() create a new dfs file.


.. code-block:: python

   from mikeio import generic
   generic.concat()