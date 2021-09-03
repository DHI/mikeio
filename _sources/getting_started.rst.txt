.. _getting_started:

Getting started
###############

Resources
*********

* Online book: `Getting started with Dfs files in Python using MIKE IO <https://dhi.github.io/getting-started-with-mikeio>`_
* `Example notebooks <https://nbviewer.jupyter.org/github/DHI/mikeio/tree/main/notebooks/>`_
* `Unit tests <https://github.com/DHI/mikeio/tree/main/tests>`_
* `DFS file system specification <https://docs.mikepoweredbydhi.com/core_libraries/dfs/dfs-file-system/>`_


Dataset
*******
The `Dataset <api.html#mikeio.Dataset>`_ is the common MIKE IO data structure for data read from dfs files. 
All `read()` methods in MIKE IO returns a Dataset with the three main properties:

* items - a list of `ItemInfo <api.html#mikeio.eum.ItemInfo>`_ with name, type and unit of each item
* time - a pandas.DateTimeIndex with the time instances of the data
* data - a list of NumPy arrays---one for each item

Read more on the `Understanding Dataset page <understanding_dataset.html>`_.


Common dfs functionality
************************
All Dfs classes and the Dataset class are representations of timeseries and 
share these properties: 

* items - a list of `ItemInfo <api.html#mikeio.eum.ItemInfo>`_ with name, type and unit of each item
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
The dfs items in MIKE IO are represented by the `ItemInfo class <api.html#mikeio.eum.ItemInfo>`_. 
An ItemInfo consists of:

* name - a user-defined string 
* type - an `EUMType <api.html#mikeio.eum.EUMType>`_ 
* unit - an `EUMUnit <api.html#mikeio.eum.EUMUnit>`_

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
A dfs0 file is also called a time series file. The MIKE IO `Dfs0 class <api.html#mikeio.Dfs0>`_ provide functionality for working with dfs0 files.  

Working with data from dfs0 files are conveniently done in one of two ways:

* mikeio.Dataset - keeps EUM information (convenient if you save data to new dfs0 file)
* pandas.DataFrame - utilize all the powerful methods of pandas


Read Dfs0 to Dataset
--------------------

.. code-block:: python

    >>> from mikeio import Dfs0
    >>> dfs = Dfs0("testdata/da_diagnostic.dfs0")
    >>> ds = dfs.read()   
   

From Dfs0 to pandas DataFrame
-----------------------------

.. code-block:: python

    >>> dfs = Dfs0("testdata/da_diagnostic.dfs0")
    >>> df = dfs.to_dataframe()

From pandas DataFrame to Dfs0
-----------------------------

.. code-block:: python

    >>> df = pd.read_csv("co2-mm-mlo.csv", parse_dates=True, index_col='Date', na_values=-99.99)
    >>> df.to_dfs0("mauna_loa_co2.dfs0")


Example notebooks
-----------------
* `Dfs0 <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs0%20-%20Timeseries.ipynb>`_ - read, write, to_dataframe, non-equidistant, accumulated timestep, extrapolation
* `Dfs0-Relative-time <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs0%20-%20Relative%20time.ipynb>`_ - read file with relative time axis
* `Dfs0 | getting-started-with-mikeio <https://dhi.github.io/getting-started-with-mikeio/dfs0.html>`_

Dfs2
****
A dfs2 file is also called a grid series file. Values in a dfs2 file are ‘element based’, i.e. values are defined in the centre of each grid cell. 
The MIKE IO `Dfs2 class <api.html#mikeio.Dfs2>`_ provide functionality for working with dfs2 files.  

.. code-block:: python

    >>> from mikeio import Dfs2
    >>> dfs = Dfs2("testdata/gebco_sound.dfs2")
    <mikeio.Dfs2>
    dx: 0.00417
    dy: 0.00417
    Items:
    0:  Elevation <Total Water Depth> (meter)
    Time: time-invariant file (1 step)   

Example notebooks
-----------------
* `Dfs2-Bathymetry <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs2%20-%20Bathymetry.ipynb>`_ - GEBCO NetCDF/xarray to dfs2 
* `Dfs2-Boundary <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs2%20-%20Boundary.ipynb>`_ - Vertical transect dfs2, interpolation in time 
* `Dfs2-Export-to-netCDF <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs2%20-%20Export%20to%20netcdf.ipynb>`_ Export dfs2 to NetCDF
* `Dfs2-GFS <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs2%20-%20Global%20Forecasting%20System.ipynb>`_ - GFS NetCDF/xarray to dfs2
* `Dfs2-SST <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs2%20-%20Sea%20surface%20temperature.ipynb>`_ - DMI NetCDF/xarray to dfs2 
* `Dfs2 | getting-started-with-mikeio <https://dhi.github.io/getting-started-with-mikeio/dfs2.html>`_


Generic dfs
***********
MIKE IO has `generic dfs <api.html#module-mikeio.generic>`_ functionality that works for all dfs files: 

* `read() <api.html#mikeio.read>`_ - Read all data to a Dataset
* `concat() <api.html#mikeio.generic.extract>`_ - Concatenates files along the time axis
* `extract() <api.html#mikeio.generic.extract>`_ - Extract timesteps and/or items to a new dfs file
* `diff() <api.html#mikeio.generic.diff>`_ - Calculate difference between two dfs files
* `sum() <api.html#mikeio.generic.extract>`_ - Calculate the sum of two dfs files
* `scale() <api.html#mikeio.generic.extract>`_ - Apply scaling to any dfs file

All methods except read() create a new dfs file.


.. code-block:: python

   from mikeio import generic
   generic.concat(["fileA.dfs2", "fileB.dfs2"], "new_file.dfs2")

.. code-block:: python

   import mikeio 
   ds = mikeio.read("new_file.dfs2")

See the `Generic notebook <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Generic.ipynb>`_ for more examples.
