.. _dfs0:

Dfs0
****

A dfs0 file is also called a time series file. The MIKE IO `Dfs0 class <dfs0.html#mikeio.Dfs0>`_ provide functionality for working with dfs0 files.  

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


Dfs0 example notebooks
----------------------
* `Dfs0 <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs0%20-%20Timeseries.ipynb>`_ - read, write, to_dataframe, non-equidistant, accumulated timestep, extrapolation
* `Dfs0-Relative-time <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs0%20-%20Relative%20time.ipynb>`_ - read file with relative time axis
* `Dfs0 | getting-started-with-mikeio <https://dhi.github.io/getting-started-with-mikeio/dfs0.html>`_



Dfs0 API
--------
.. autoclass:: mikeio.Dfs0
	:members:
	:inherited-members:

