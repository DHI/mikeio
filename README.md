
![logo](https://raw.githubusercontent.com/DHI/mikeio/main/images/logo/PNG/MIKE-IO-Logo-Pos-RGB-nomargin.png)
# MIKE IO: input/output of MIKE files in Python
 ![Python version](https://img.shields.io/pypi/pyversions/mikeio.svg)
 [![Full test](https://github.com/DHI/mikeio/actions/workflows/full_test.yml/badge.svg)](https://github.com/DHI/mikeio/actions/workflows/full_test.yml)
[![PyPI version](https://badge.fury.io/py/mikeio.svg)](https://badge.fury.io/py/mikeio)


Read, write and manipulate dfs0, dfs1, dfs2, dfs3, dfsu and mesh files.

MIKE IO facilitates common data processing workflows for [MIKE files in Python](https://www.mikepoweredbydhi.com/products/mike-for-developers#io).


[![YouTube](images/youtube1.png)](http://www.youtube.com/watch?v=Jm0iAeK8QW0)

## MIKE IO 1.0
MIKE IO 1.0 is now available and it has a lot of benefits to working with dfs files even easier than before.

[![YouTube](images/youtube2.png)](http://www.youtube.com/watch?v=0oVedpx9zAQ)


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
* General help, new ideas and feature requests - [GitHub Discussions](http://github.com/DHI/mikeio/discussions) 
* Bugs - [GitHub Issues](http://github.com/DHI/mikeio/issues) 

## Installation

From PyPI: 

`pip install mikeio`

Or development version:

`pip install https://github.com/DHI/mikeio/archive/main.zip`


## Tested

MIKE IO is tested extensively.

See detailed test coverage report below:
```
---------- coverage: platform linux, python 3.10.4-final-0 -----------
Name                              Stmts   Miss  Cover
-----------------------------------------------------
mikeio/__init__.py                   39      2    95%
mikeio/base.py                       26      5    81%
mikeio/custom_exceptions.py          25      8    68%
mikeio/data_utils.py                127     22    83%
mikeio/dataarray.py                 927    145    84%
mikeio/dataset.py                   722     92    87%
mikeio/dfs0.py                      284     33    88%
mikeio/dfs1.py                       62      6    90%
mikeio/dfs2.py                      244     42    83%
mikeio/dfs3.py                      201     15    93%
mikeio/dfs.py                       275     30    89%
mikeio/dfsu.py                      699     63    91%
mikeio/dfsu_factory.py               41      2    95%
mikeio/dfsu_layered.py              186     23    88%
mikeio/dfsu_spectral.py             128      7    95%
mikeio/dfsutil.py                   104      8    92%
mikeio/eum.py                      1297      3    99%
mikeio/generic.py                   396      9    98%
mikeio/helpers.py                    16      5    69%
mikeio/interpolation.py              63      1    98%
mikeio/pfs.py                        93      0   100%
mikeio/spatial/FM_geometry.py      1123    116    90%
mikeio/spatial/FM_utils.py          293     30    90%
mikeio/spatial/__init__.py            0      0   100%
mikeio/spatial/crs.py                50     25    50%
mikeio/spatial/geometry.py           88     33    62%
mikeio/spatial/grid_geometry.py     497     33    93%
mikeio/spatial/utils.py              38      0   100%
mikeio/spectral_utils.py             89      5    94%
mikeio/xyz.py                        12      0   100%
-----------------------------------------------------
TOTAL                              8145    763    91%


============ 561 passed in 74.58s (0:01:14) ============
```

## Cloud enabled

It is possible to run MIKE IO in your favorite cloud notebook environment e.g. [Deepnote](https://deepnote.com/), [Google Colab](https://colab.research.google.com/), etc...

![DeepNote](images/deepnote.png)

![Colab](images/colab.png)


