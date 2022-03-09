Getting started
###############

Resources
*********

* Online book: `Getting started with Dfs files in Python using MIKE IO <https://dhi.github.io/getting-started-with-mikeio>`_
* Online book: `Python for marine modelers using MIKE IO and FMskill <https://dhi.github.io/book-learn-mikeio-fmskill>`_
* `Example notebooks <https://nbviewer.jupyter.org/github/DHI/mikeio/tree/main/notebooks/>`_
* `Unit tests <https://github.com/DHI/mikeio/tree/main/tests>`_
* `DFS file system specification <https://docs.mikepoweredbydhi.com/core_libraries/dfs/dfs-file-system/>`_


Dataset
*******
The `Dataset <dataset.html#mikeio.Dataset>`_ is the common MIKE IO data structure for data read from dfs files. 
The  `mikeio.read()` method returns a Dataset with a DataArray for each item.

The DataArray have all the relevant information, e.g:

* item - an `ItemInfo <eum.html#mikeio.eum.ItemInfo>`_ with name, type and unit
* time - a pandas.DateTimeIndex with the time instances of the data
* values - a NumPy array

Read more on the `Dataset page <dataset.html>`_.


Common dfs functionality
************************
All Dfs classes and the Dataset class are representations of timeseries and 
share these properties: 

* items - a list of `ItemInfo <eum.html#mikeio.eum.ItemInfo>`_ with name, type and unit of each item
* n_items - Number of items
* n_timesteps - Number of timesteps
* start_time - First time instance (as datetime)
* end_time - Last time instance (as datetime)
* deletevalue - File delete value (NaN value)

All Dfs classes have these methods:

* read(items, time_steps, ...)
* write(filename, data, ...)


Items, ItemInfo and EUM
***********************
The dfs items in MIKE IO are represented by the `ItemInfo class <eum.html#mikeio.eum.ItemInfo>`_. 
An ItemInfo consists of:

* name - a user-defined string 
* type - an `EUMType <eum.html#mikeio.eum.EUMType>`_ 
* unit - an `EUMUnit <eum.html#mikeio.eum.EUMUnit>`_

.. code-block:: python

    >>> from mikeio.eum import ItemInfo, EUMType
    >>> item = ItemInfo("Viken", EUMType.Water_Level)
    >>> item
    Viken <Water Level> (meter)
    >>> ItemInfo(EUMType.Wind_speed)
    Wind speed <Wind speed> (meter per sec)

See the `Units notebook <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Units.ipynb>`_ for more examples.


Dfs0
****
A dfs0 file is also called a time series file. The MIKE IO `Dfs0 class <dfs0.html#mikeio.Dfs0>`_ provide functionality for working with dfs0 files.  

Working with data from dfs0 files are conveniently done in one of two ways:

* mikeio.Dataset - keeps EUM information (convenient if you save data to new dfs0 file)
* pandas.DataFrame - utilize all the powerful methods of pandas


Read Dfs0 to Dataset:

.. code-block:: python

    >>> from mikeio import Dfs0
    >>> dfs = Dfs0("testdata/da_diagnostic.dfs0")
    >>> ds = dfs.read()   

Read more on the `Dfs0 page <dfs0.html>`_.



Dfs2
****
A dfs2 file is also called a grid series file. Values in a dfs2 file are ‘element based’, i.e. values are defined in the centre of each grid cell. 
The MIKE IO `Dfs2 class <dfs123.html#mikeio.Dfs2>`_ provide functionality for working with dfs2 files.  

.. code-block:: python

    >>> from mikeio import Dfs2
    >>> dfs = Dfs2("testdata/gebco_sound.dfs2")
    <mikeio.Dfs2>
    dx: 0.00417
    dy: 0.00417
    Items:
    0:  Elevation <Total Water Depth> (meter)
    Time: time-invariant file (1 step)   

Read more on the `Dfs123 page <dfs123.html>`_


Generic dfs
***********
MIKE IO has `generic dfs <generic.html#module-mikeio.generic>`_ functionality that works for all dfs files: 

* `read() <generic.html#mikeio.read>`_ - Read all data to a Dataset
* `concat() <generic.html#mikeio.generic.extract>`_ - Concatenates files along the time axis
* `extract() <generic.html#mikeio.generic.extract>`_ - Extract timesteps and/or items to a new dfs file
* `diff() <generic.html#mikeio.generic.diff>`_ - Calculate difference between two dfs files
* `sum() <generic.html#mikeio.generic.extract>`_ - Calculate the sum of two dfs files
* `scale() <generic.html#mikeio.generic.extract>`_ - Apply scaling to any dfs file
* `avg_time() <generic.html#mikeio.generic.avg_time>`_ - Create a temporally averaged dfs file
* `quantile() <generic.html#mikeio.generic.quantile>`_ - Create a dfs file with temporal quantiles

All methods except read() create a new dfs file.

.. code-block:: python

   from mikeio import generic
   generic.concat(["fileA.dfs2", "fileB.dfs2"], "new_file.dfs2")

.. code-block:: python

   import mikeio 
   ds = mikeio.read("new_file.dfs2")

See `Generic page <generic.html>`_ and the `Generic notebook <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Generic.ipynb>`_ for more examples.
