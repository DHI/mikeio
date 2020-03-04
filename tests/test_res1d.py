from mikeio.res1d import Res1D, ExtractionPoint


def test_read_single_item():
    file = r"testdata/Exam6Base.res1d"

    p1 = ExtractionPoint()
    p1.BranchName = '104l1'
    p1.Chainage = 34.4131
    p1.VariableType = 'WaterLevel'
    ts = Res1D().read(file, [p1])

    assert len(ts) == 110
