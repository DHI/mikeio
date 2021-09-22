.. _generic:

Generic
*******

MIKE IO has `generic dfs <#module-mikeio.generic>`_ functionality that works for all dfs files: 

* `read() <#mikeio.read>`_ - Read all data to a Dataset
* `concat() <#mikeio.generic.extract>`_ - Concatenates files along the time axis
* `extract() <#mikeio.generic.extract>`_ - Extract timesteps and/or items to a new dfs file
* `diff() <#mikeio.generic.diff>`_ - Calculate difference between two dfs files
* `sum() <#mikeio.generic.extract>`_ - Calculate the sum of two dfs files
* `scale() <#mikeio.generic.extract>`_ - Apply scaling to any dfs file

All methods except read() create a new dfs file.

.. code-block:: python

   from mikeio import generic
   generic.concat(["fileA.dfs2", "fileB.dfs2"], "new_file.dfs2")

.. code-block:: python

   import mikeio 
   ds = mikeio.read("new_file.dfs2")

See the `Generic notebook <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Generic.ipynb>`_ for more examples.





Generic API
-----------
.. automodule:: mikeio
	:members:

.. automodule:: mikeio.generic
	:members:
