.. _intro:

Requirements
------------

* Windows or Linux operating system
* Python x64 3.6 - 3.9
* (Windows) `VC++ redistributables <https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>`_ (already installed if you have MIKE)

Installation
------------
From PyPI::

    pip install mikeio

Or development version::

    pip install https://github.com/DHI/mikeio/archive/main.zip

Getting started
---------------
    
    >>>  from mikeio import Dfs0
    >>>  dfs = Dfs0('simple.dfs0')
    >>>  df = dfs.to_dataframe()

Notebooks
---------

* `Dfs0 <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs0%20-%20Timeseries.ipynb>`_
* `Dfsu basic <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfsu%20-%20Read.ipynb>`_
* `Create Dfs2 from netCDF <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs2%20-%20Bathymetry.ipynb>`_
* `Complete list of all notebooks <https://nbviewer.jupyter.org/github/DHI/mikeio/tree/main/notebooks/>`_

Where can I get help?
---------------------

* New ideas and feature requests - `GitHub Discussions <http://github.com/DHI/mikeio/discussions>`_ 
* Bugs - `GitHub Issues <http://github.com/DHI/mikeio/issues>`_