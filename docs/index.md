
# MIKE IO: input/output of MIKE files in Python

![](../images/logo/SVG/MIKE-IO-Logo-Pos-RGB.svg)

Read, write and manipulate dfs0, dfs1, dfs2, dfs3, dfsu and mesh files. 

See our sister library [MIKE IO 1D](https://github.com/DHI/mikeio1d) for .res1d and .xns11 files.

**The documentation is currently being updated to reflect the upcoming version of MIKE IO 1.0**

## Requirements

* Windows or Linux operating system
* Python x64 3.7 - 3.10
* (Windows) [VC++ redistributables](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>) (already installed if you have MIKE)

## Installation

```
$ pip install mikeio
```

## Getting started

```python
>>>  import
>>>  ds = mikeio.read('simple.dfs0')
>>>  df = ds.to_dataframe()
```

Read more in the [getting started guide](getting_started).


Where can I get help?
---------------------

* New ideas and feature requests - [GitHub Discussions](https://github.com/DHI/mikeio/discussions)
* Bugs - [GitHub Issues](https://github.com/DHI/mikeio/issues)

```{eval-rst}
.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   getting_started
   design
   data_structures
   dataset
   dataarray   
   dfs0
   dfs123
   dfsu   
   eum
   generic
```