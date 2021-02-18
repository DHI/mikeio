.. mikeio documentation master file, created by
   sphinx-quickstart on Thu Apr  2 12:42:36 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://raw.githubusercontent.com/DHI/mikeio/master/images/logo/SVG/MIKE-IO-Logo-Pos-RGB.svg

MIKE IO: input/output of MIKE files in Python
=============================================

Facilitate creating, reading and writing dfs0, dfs2, dfs1 and dfs3, dfsu and mesh files. Reading Res1D data.

Requirements
------------

* Windows operating system
* Python x64 3.6, 3.7 or 3.8 
* `VC++ redistributables <https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>`_ (already installed if you have MIKE)

Installation
------------
From PyPI: 

    pip install mikeio

For Anaconda:

    conda install -c conda-forge mikeio

Or development version:

    pip install https://github.com/DHI/mikeio/archive/master.zip

Getting started
---------------
    
    >>>  from mikeio import Dfs0
    >>>  dfs = Dfs0('simple.dfs0')
    >>>  df = dfs.to_dataframe()

Notebooks
---------

* `Dfs0 <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/master/notebooks/Dfs0%20-%20Timeseries.ipynb>`_
* `Dfsu basic <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/master/notebooks/Dfsu%20-%20Read.ipynb>`_
* `Create Dfs2 from netCDF <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/master/notebooks/Dfs2%20-%20Bathymetry.ipynb>`_
* `Complete list of all notebooks <https://nbviewer.jupyter.org/github/DHI/mikeio/tree/master/notebooks/>`_

Where can I get help?
---------------------

* New ideas and feature requests - `GitHub Discussions <http://github.com/DHI/mikeio/discussions>`_ 
* Bugs - `GitHub Issues <http://github.com/DHI/mikeio/issues>`_


.. toctree::
   api
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
