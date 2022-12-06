# content of conftest.py
import pytest
import numpy
import pandas
import mikeio


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["np"] = numpy

@pytest.fixture(autouse=True)
def add_pd(doctest_namespace):
    doctest_namespace["pd"] = pandas


@pytest.fixture(autouse=True)
def add_mikeio(doctest_namespace):
    doctest_namespace["mikeio"] = mikeio