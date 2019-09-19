from pydhi.helpers import safe_length

import clr

clr.AddReference('System.Collections')
from System import Int32
from System.Collections.Generic import List

def test_safe_length_returns_length_for_python_list():

    a = [1,2,6]

    n = safe_length(a)

    assert n==3

def test_safe_length_returns_length_for_net_list():

    a = List[Int32]()
    a.Add(1)
    a.Add(2)
    a.Add(6)

    n = safe_length(a)

    assert n==3