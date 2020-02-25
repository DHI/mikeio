import os
import numpy as np
import datetime
from mikeio import res1d as r1d

def test_read_single_item():

    file = r"testdata/Exam6Base.res1d"

    p1 = r1d.ExtractionPoint()
    p1.BranchName = '104l1'
    p1.Chainage = 34.4131
    p1.VariableType = 'WaterLevel'
    ts = r1d.res1d().read(file, [p1])

    assert len(ts) == 110