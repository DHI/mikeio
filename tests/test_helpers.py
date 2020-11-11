import pytest
from mikeio.helpers import safe_length, to_datatype
from mikeio.custom_exceptions import InvalidDataValueType

import clr

clr.AddReference("System.Collections")
from System import Int32
from System.Collections.Generic import List


def test_safe_length_returns_length_for_python_list():

    a = [1, 2, 6]

    n = safe_length(a)

    assert n == 3


def test_safe_length_returns_length_for_net_list():

    a = List[Int32]()
    a.Add(1)
    a.Add(2)
    a.Add(6)

    n = safe_length(a)

    assert n == 3


def test_to_data_value_type():
    assert to_datatype(0) == to_datatype("Instantaneous")
    assert to_datatype(1) == to_datatype("Accumulated")
    assert to_datatype(2) == to_datatype("StepAccumulated")
    assert to_datatype(3) == to_datatype("MeanStepBackward")
    assert to_datatype(4) == to_datatype("MeanStepForward")


def test_invalid_input():
    with pytest.raises(InvalidDataValueType):
        to_datatype("Not an acceptable string")
