
![logo](https://raw.githubusercontent.com/DHI/mikeio/main/images/logo/PNG/MIKE-IO-Logo-Pos-RGB-nomargin.png)
# MIKE IO: input/output of MIKE files in Python
 ![Python version](https://img.shields.io/pypi/pyversions/mikeio.svg)
 [![Full test](https://github.com/DHI/mikeio/actions/workflows/full_test.yml/badge.svg)](https://github.com/DHI/mikeio/actions/workflows/full_test.yml)
[![PyPI version](https://badge.fury.io/py/mikeio.svg)](https://badge.fury.io/py/mikeio)
![OS](https://img.shields.io/badge/OS-Windows%20%7C%20Linux-blue)
![Downloads](https://img.shields.io/pypi/dm/mikeio)



Read, write and manipulate dfs0, dfs1, dfs2, dfs3, dfsu and mesh files.

MIKE IO facilitates common data processing workflows for [MIKE files](https://www.dhigroup.com/technologies/mikepoweredbydhi) using Python.

[![MIKEIO. Read, write and analyze MIKE dfs files with Python on Vimeo](https://raw.githubusercontent.com/DHI/mikeio/main/images/youtube1.png)](https://player.vimeo.com/video/708275619)

<!--[![MIKEIO. New workflow and data structures in MIKE IO 1.0 on Vimeo](https://raw.githubusercontent.com/DHI/mikeio/main/images/youtube2.png)](https://player.vimeo.com/video/708276337)-->


## Requirements
* Windows or Linux operating system
* Python x64 3.10 - 3.13
* (Windows) [VC++ redistributables](https://aka.ms/vs/17/release/vc_redist.x64.exe) (already installed if you have MIKE)

## Installation

From PyPI: 

`pip install mikeio`

Or development version:

`pip install https://github.com/DHI/mikeio/archive/main.zip`

:warning: **Don't use conda to install MIKE IO!**, the version on conda is outdated.

## Getting started

The material from the last Academy by DHI course is available here: [Getting started with Dfs files in Python using MIKE IO](https://dhi.github.io/getting-started-with-mikeio/intro.html)

## Where can I get help?
* Documentation - [https://dhi.github.io/mikeio/](https://dhi.github.io/mikeio/)
* General help, new ideas and feature requests - [GitHub Discussions](http://github.com/DHI/mikeio/discussions) 
* Bugs - [GitHub Issues](http://github.com/DHI/mikeio/issues) 


## Tested

MIKE IO is tested extensively.

See detailed test coverage report below:
```bash
$ pytest --cov=mikeio
```
<pre>
---------- coverage: platform linux, python 3.13.0-final-0 -----------
Name                                      Stmts   Miss  Cover
-------------------------------------------------------------
mikeio/__init__.py                           33      5    85%
mikeio/_interpolation.py                     68      6    91%
mikeio/_spectral.py                          97      7    93%
mikeio/_time.py                              28      1    96%
mikeio/_track.py                            119      9    92%
mikeio/dataset/__init__.py                    4      0   100%
mikeio/dataset/_data_plot.py                359     38    89%
mikeio/dataset/_data_utils.py                19      0   100%
mikeio/dataset/_dataarray.py                725     46    94%
mikeio/dataset/_dataset.py                  766     51    93%
mikeio/dfs/__init__.py                        5      0   100%
mikeio/dfs/_dfs0.py                         205     13    94%
mikeio/dfs/_dfs1.py                          89      2    98%
mikeio/dfs/_dfs2.py                         142      3    98%
mikeio/dfs/_dfs3.py                         157     11    93%
mikeio/dfs/_dfs.py                          251     14    94%
mikeio/dfsu/__init__.py                       6      0   100%
mikeio/dfsu/_common.py                       36      1    97%
mikeio/dfsu/_dfsu.py                        239      6    97%
mikeio/dfsu/_factory.py                      20      1    95%
mikeio/dfsu/_layered.py                     204      9    96%
mikeio/dfsu/_mesh.py                         54      8    85%
mikeio/dfsu/_spectral.py                    234     42    82%
mikeio/eum/__init__.py                        2      0   100%
mikeio/eum/_eum.py                         1353      9    99%
mikeio/exceptions.py                         24      4    83%
mikeio/generic.py                           475     13    97%
mikeio/pfs/__init__.py                        7      0   100%
mikeio/pfs/_pfsdocument.py                  233      4    98%
mikeio/pfs/_pfssection.py                   225     11    95%
mikeio/spatial/_FM_geometry.py              511     13    97%
mikeio/spatial/_FM_geometry_layered.py      417     30    93%
mikeio/spatial/_FM_geometry_spectral.py      94      9    90%
mikeio/spatial/_FM_utils.py                 306     22    93%
mikeio/spatial/__init__.py                    6      0   100%
mikeio/spatial/_geometry.py                 100      8    92%
mikeio/spatial/_grid_geometry.py            629     41    93%
mikeio/spatial/_utils.py                     38      0   100%
mikeio/spatial/crs.py                        51      5    90%
-------------------------------------------------------------
TOTAL                                      8331    442    95%
</pre>

## Cloud enabled

It is possible to run MIKE IO in your favorite cloud notebook environment e.g. [Deepnote](https://deepnote.com/), [Google Colab](https://colab.research.google.com/), etc...

![DeepNote](https://raw.githubusercontent.com/DHI/mikeio/main/images/deepnote.png)

![Colab](https://raw.githubusercontent.com/DHI/mikeio/main/images/colab.png)


