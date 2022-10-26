
![](../images/logo/SVG/MIKE-IO-Logo-Pos-RGB.svg)

# MIKE IO: input/output of MIKE files in Python
 ![Python version](https://img.shields.io/pypi/pyversions/mikeio.svg)
[![PyPI version](https://badge.fury.io/py/mikeio.svg)](https://badge.fury.io/py/mikeio)

Read, write and manipulate dfs0, dfs1, dfs2, dfs3, dfsu and mesh files. 

See our sister library [MIKE IO 1D](https://github.com/DHI/mikeio1d) for .res1d and .xns11 files.

## Requirements

* Windows or Linux operating system
* Python x64 3.8 - 3.11
* (Windows) [VC++ redistributables](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>) (already installed if you have MIKE)

## Installation

```
$ pip install mikeio
```
**Don't use conda to install MIKE IO!**, the version on conda is outdated.

## Getting started

```python
>>>  import
>>>  ds = mikeio.read('simple.dfs0')
>>>  df = ds.to_dataframe()
```

Read more in the [getting started guide](getting-started).


Where can I get help?
---------------------

* New ideas and feature requests - [GitHub Discussions](https://github.com/DHI/mikeio/discussions)
* Bugs - [GitHub Issues](https://github.com/DHI/mikeio/issues)

```{eval-rst}
.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   getting-started
   design
   data-structures
   dataset
   dataarray
   dfs-overview
   dfs0
   dfs1
   dfs2
   dfs3
   dfsu-mesh-overview
   mesh
   dfsu-2d
   dfsu-3d
   dfsu-2dv-vertical-profile
   dfsu-spectral
   eum
   pfs
   generic
```
