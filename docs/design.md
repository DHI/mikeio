# Design philosophy

* Easy to use
* Familiar
* Easy to install
* Easy to get started
* Open Source​
* Easy to collaborate​
* Reproducible
* Easy access to new features


## Easy to use

Common operations such as reading a file should only need a few lines of code.

Make extensive use of existing standard libraries for scientific computing such as numpy, matplotlib and pandas.


## Familiar

MIKE IO aims to use a syntax familiar to users of scientific computing libraries such as NumPy, Pandas and xarray.

## Easy to install

From PyPI::

    pip install mikeio

## Easy to get started
By providing many examples to cut/paste from.

Examples are available in two forms:

* [Unit tests](https://github.com/DHI/mikeio/tree/main/tests)
* [Jupyter notebooks](https://nbviewer.jupyter.org/github/DHI/mikeio/tree/main/notebooks/)

## Open Source​
MIKE IO is an open source project licensed under the [BSD-3 license](https://github.com/DHI/mikeio/blob/main/License.txt).
The software is provided free of charge with the source code available for inspection and modification.

Contributions are welcome, more details can be found in our [contribution guidelines](https://github.com/DHI/mikeio/blob/main/CONTRIBUTING.md).

## Easy to collaborate
By developing MIKE IO on GitHub along with a completely open discussion, we believe that the collaboration between developers and end-users results in a useful library.

## Reproducible
By providing the historical versions of MIKE IO on PyPI it is possible to reproduce the behaviour of an older existing system, based on an older version.

Install specific version::

```
pip install mikeio==0.12.2
```

## Easy access to new features
Features are being added all the time, by developers at DHI in offices all around the globe as well as external contributors using MIKE IO in their work.
These new features are always available from the main branch on GitHub and thanks to automated testing, it is always possible to verify that the tests passes before downloading a new development version.

Install development version::

```
pip install https://github.com/DHI/mikeio/archive/main.zip
```
