from mikeio.xns11 import Xns11, ExtractionPoint


def test_read():
    xns11 = Xns11()

    query = ExtractionPoint()
    query.BranchName = 'BigCreek'
    query.TopoId = 'baseline'
    query.Chainage = 741.71

    geometry = xns11.read("tests/testdata/x_sections.xns11", [query])

    assert geometry is not None
