
![logo](https://raw.githubusercontent.com/DHI/mikeio/main/images/logo/PNG/MIKE-IO-Logo-Pos-RGB-nomargin.png)
# MIKE IO: input/output of MIKE files in python
 ![Python version](https://img.shields.io/pypi/pyversions/mikeio.svg)
 [![Full test](https://github.com/DHI/mikeio/actions/workflows/full_test.yml/badge.svg)](https://github.com/DHI/mikeio/actions/workflows/full_test.yml)
[![PyPI version](https://badge.fury.io/py/mikeio.svg)](https://badge.fury.io/py/mikeio)


Read, write and manipulate dfs0, dfs1, dfs2, dfs3, dfsu and mesh files.

Facilitates common data processing workflows for MIKE files.

*For res1d and xns11 files use the related package [MIKE IO 1D](https://github.com/DHI/mikeio1d)*

## Upcoming release: MIKE IO 1.0
MIKE IO 1.0 is planned to be released in May 2022 and it will have a lot of benefits to make working with dfs files easier, but it also requires some changes to your existing code. More details in the [discussion page](https://github.com/DHI/mikeio/discussions/279).

![code example](https://raw.githubusercontent.com/DHI/mikeio/main/images/code.png)

### Important changes
* New class `mikeio.DataArray` which will be the main class to interact with, having these properties and methods
  - item info
  - geometry (grid coordinates)
  - methods for plotting
  - methods for aggreation in time and space
* Indexing into a dataset e.g. `ds.Surface_elevation` to get a specific item, will not return a numpy array, but a `mikeio.DataArray`

## Requirements
* Windows or Linux operating system
* Python x64 3.7 - 3.10
* (Windows) [VC++ redistributables](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) (already installed if you have MIKE)

[More info about dependencies](http://docs.mikepoweredbydhi.com/nuget/)

## Where can I get help?
* Documentation - [https://dhi.github.io/mikeio/](https://dhi.github.io/mikeio/)
* New ideas and feature requests - [GitHub Discussions](http://github.com/DHI/mikeio/discussions) 
* Bugs - [GitHub Issues](http://github.com/DHI/mikeio/issues) 
* General help, FAQ - [Stackoverflow with the tag `mikeio`](https://stackoverflow.com/questions/tagged/mikeio)

## Installation

From PyPI: 

`pip install mikeio`

Or development version:

`pip install https://github.com/DHI/mikeio/archive/main.zip`


## Tested

MIKE IO is tested extensively. **95%** total test coverage.

See detailed test coverage report below:
```
---------- coverage: platform linux, python 3.10.2-final-0 -----------
Name                              Stmts   Miss  Cover
-----------------------------------------------------
mikeio/__init__.py                   38      3    92%
mikeio/aggregator.py                 98      9    91%
mikeio/base.py                       26      5    81%
mikeio/custom_exceptions.py          25      6    76%
mikeio/data_utils.py                111     24    78%
mikeio/dataarray.py                 686    101    85%
mikeio/dataset.py                   695     87    87%
mikeio/dfs0.py                      278     26    91%
mikeio/dfs1.py                       61      6    90%
mikeio/dfs2.py                      186     37    80%
mikeio/dfs3.py                      202     77    62%
mikeio/dfs.py                       269     21    92%
mikeio/dfsu.py                      735     56    92%
mikeio/dfsu_factory.py               41      2    95%
mikeio/dfsu_layered.py              142     19    87%
mikeio/dfsu_spectral.py              97      8    92%
mikeio/dfsutil.py                    89      5    94%
mikeio/eum.py                      1297      4    99%
mikeio/generic.py                   399      8    98%
mikeio/helpers.py                    16      5    69%
mikeio/interpolation.py              63      2    97%
mikeio/pfs.py                        95      0   100%
mikeio/spatial/FM_geometry.py       867     80    91%
mikeio/spatial/FM_utils.py          231     19    92%
mikeio/spatial/__init__.py            4      0   100%
mikeio/spatial/crs.py                50     25    50%
mikeio/spatial/geometry.py           88     34    61%
mikeio/spatial/grid_geometry.py     334     16    95%
mikeio/spatial/spatial.py           278    181    35%
mikeio/xyz.py                        12      0   100%
-----------------------------------------------------
TOTAL                              7513    866    88%

================ 454 passed in 41.76s =================
```

## Cloud enabled

From MIKE IO v.0.7 it is possible to run MIKE IO in your favorite cloud notebook environment e.g. [Google Colab](https://colab.research.google.com/), [DeepNote](https://deepnote.com/), etc...

![DeepNote](images/deepnote.png)

![Colab](images/colab.png)


