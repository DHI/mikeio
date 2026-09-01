
![logo](https://raw.githubusercontent.com/DHI/mikeio/main/images/logo/PNG/MIKE-IO-Logo-Pos-RGB-nomargin.png)
# MIKE IO: input/output of MIKE files in Python
 ![Python version](https://img.shields.io/pypi/pyversions/mikeio.svg)
 [![Full test](https://github.com/DHI/mikeio/actions/workflows/full_test.yml/badge.svg)](https://github.com/DHI/mikeio/actions/workflows/full_test.yml)
[![PyPI version](https://badge.fury.io/py/mikeio.svg)](https://badge.fury.io/py/mikeio)
![OS](https://img.shields.io/badge/OS-Windows%20%7C%20Linux-blue)
![Downloads](https://img.shields.io/pypi/dm/mikeio)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://github.com/DHI/mikeio/blob/main/License.txt)

Read, write and manipulate dfs0, dfs1, dfs2, dfs3, dfsu and mesh files.

MIKE IO facilitates common data processing workflows for [MIKE files](https://www.dhigroup.com/technologies/mikepoweredbydhi) using Python.

## Requirements

* Windows or Linux operating system
* Python x64 3.12 - 3.14
* (Windows) [VC++ redistributables](https://aka.ms/vs/17/release/vc_redist.x64.exe) (already installed if you have MIKE)

## Installation

```bash
pip install mikeio
```

:warning: **Don't use conda to install MIKE IO!** The version on conda-forge is several major versions behind.

## Getting started

Read a file into a `Dataset` — a collection of named `DataArray`s sharing time and geometry:

```python
>>> import mikeio
>>> ds = mikeio.read("HD2D.dfsu")
>>> ds
<mikeio.Dataset>
title: Output 1
dims: (time:9, element:884)
time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00 (9 records)
geometry: Dfsu2D (884 elements, 529 nodes)
items:
  0:  Surface elevation <Surface Elevation> (meter)
  1:  U velocity <u velocity component> (meter per sec)
  2:  V velocity <v velocity component> (meter per sec)
  3:  Current speed <Current Speed> (meter per sec)
```

Select an item and extract a time series at a point:

```python
>>> da = ds["Surface elevation"]
>>> da.sel(x=606200, y=6905480)
<mikeio.DataArray>
name: Surface elevation
dims: (time:9)
time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00 (9 records)
geometry: GeometryPoint2D(x=606202.7806372638, y=6905474.639383219)
values: [0.4595, 0.807, ..., -0.6322]
```

Convert to pandas, or write the result back to a dfs file:

```python
>>> df = mikeio.read("da_diagnostic.dfs0").to_dataframe()
>>> ds.to_dfs("output.dfsu")
```

See the [user guide](https://dhi.github.io/mikeio/user-guide/getting-started.html) for more.

## Where can I get help?
* Documentation - [https://dhi.github.io/mikeio/](https://dhi.github.io/mikeio/)
* General help, new ideas and feature requests - [GitHub Discussions](https://github.com/DHI/mikeio/discussions)
* Bugs - [GitHub Issues](https://github.com/DHI/mikeio/issues)
* Course material: [Getting started with Dfs files in Python using MIKE IO](https://dhi.github.io/getting-started-with-mikeio/intro.html)

## Testing

MIKE IO is tested extensively, with an overall statement coverage of ~95%. The test suite runs on every pull request against Python 3.12 and 3.14, and on a schedule on both Linux and Windows.

```bash
uv run pytest --cov=mikeio
```

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](https://github.com/DHI/mikeio/blob/main/CONTRIBUTING.md). Key architectural decisions are documented as [ADRs](https://github.com/DHI/mikeio/tree/main/adr).

## License

[BSD-3-Clause](https://github.com/DHI/mikeio/blob/main/License.txt)
