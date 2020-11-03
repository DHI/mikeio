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

* Python x64 >=3.6
* MIKE >= 2019 or MIKE SDK >= 2017 

Installation
------------

    pip install mikeio

Getting started
---------------
    
    >>>  from mikeio import Dfs0
    >>>  dfs = Dfs0('simple.dfs0')
    >>>  df = dfs.to_dataframe()
    >>> df.head()
                         VarFun01    NotFun
    2017-01-01 00:00:00  0.843547  0.640486
    2017-01-01 05:00:00  0.093729  0.653257
    2017-01-01 10:00:00       NaN       NaN
    

.. toctree::
   api
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
