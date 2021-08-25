.. _understanding_dataset:

Understanding Dataset
*********************
The MIKE IO `Dataset <api.html#mikeio.Dataset>`_ is a common data structure for data read from dfs files. 
All `read()` methods in MIKE IO returns a Dataset with three main properties:

* items - a list of `ItemInfo <api.html#mikeio.eum.ItemInfo>`_ with name, type and unit of each item
* time - a pandas.DateTimeIndex with the time instances of the data
* data - a list of NumPy arrays---one for each item


Dataset selecting and subsetting
--------------------------------
TODO


Dataset properties
------------------
The Dataset has several convenience properties including:

* is_equidistant


Dataset methods
---------------
TODO
