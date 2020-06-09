from mikeio.xns11 import Xns11, QueryData


def test_read():
    file_path = "tests/testdata/x_sections.xns11"
    query = QueryData('baseline', 'BigCreek', 741.71)
    geometry = Xns11().read(file_path, [query])

    assert geometry is not None
