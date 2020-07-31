import numpy as np

from mikeio.dotnet import to_dotnet_array

def test_float_array_np_dotnet():

    x = np.random.random(10)

    netx = to_dotnet_array(x)

    assert netx.Length == 10