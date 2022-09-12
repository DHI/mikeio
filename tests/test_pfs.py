import sys
import os
import pytest

from mikeio import Pfs


def test_basic():

    pfs = Pfs("tests/testdata/simple.pfs")

    data = pfs.data
    # On a pfs file with a single target, the target is implicit,
    #  i.e. BoundaryExtractor in this case

    assert data.z_min == -3000
    assert data.POINT_1.y == 50


def test_mztoolbox():
    pfs = Pfs("tests/testdata/concat.mzt")
    assert "tide1.dfs" in pfs.data.Setup.File_1.InputFile
    assert "|" in pfs.data.Setup.File_1.InputFile


def check_txt_files_match(f1, f2, comment="//"):
    with open(f1) as file:
        file1lines = file.read().split("\n")

    with open(f2) as file:
        file2lines = file.read().split("\n")

    for j in range(len(file1lines)):
        s1 = file1lines[j].strip()
        if s1 == "" or s1.startswith(comment):
            continue

        s2 = file2lines[j].strip()
        assert s1 == s2


def test_read_write(tmpdir):
    infilename = "tests/testdata/concat.mzt"
    pfs1 = Pfs(infilename)
    outfilename = os.path.join(tmpdir.dirname, "concat_out.mzt")
    pfs1.write(outfilename)
    check_txt_files_match(infilename, outfilename)
    _ = Pfs(outfilename) # try to parse it also


def test_sw():

    pfs = Pfs("tests/testdata/lake.sw")
    data = pfs.data

    # On a pfs file with a single target, the target is implicit,
    #  i.e. FemEngineSW in this case

    assert data.SPECTRAL_WAVE_MODULE.SPECTRAL.number_of_frequencies == 25
    assert data.SPECTRAL_WAVE_MODULE.SPECTRAL.number_of_frequencies == 25

    # use shorthand alias SW instead of SPECTRAL_WAVE_MODULE
    # assert data.SW.SPECTRAL.number_of_frequencies == 25
    # assert data.SW.WIND.format == 1

    assert data.TIME.number_of_time_steps == 450

    assert data.TIME.start_time.year == 2002
    assert data.TIME.start_time.month == 1


def test_outputs():

    pfs = Pfs("tests/testdata/lake.sw")
    df = pfs.get_outputs(section="SPECTRAL_WAVE_MODULE")

    assert df["file_name"][1] == "Wave_parameters.dfsu"


# def test_sw_outputs():

#     pfs = Pfs("tests/testdata/lake.sw")
#     df = pfs.data.SW.get_outputs()

#     assert df["file_name"][1] == "Wave_parameters.dfsu"
#     assert df.shape[0] == 4

#     df = pfs.data.SW.get_outputs(included_only=True)

#     assert df["file_name"][1] == "Wave_parameters.dfsu"
#     assert df.shape[0] == 3


# def test_hd_outputs():

#     pfs = Pfs("tests/testdata/lake.m21fm")
#     df = pfs.data.HD.get_outputs()

#     assert df["file_name"][2] == "ts.dfs0"
#     assert df.shape[0] == 3

#     df = pfs.data.HD.get_outputs(included_only=True)

#     assert df.shape[0] == 2


def test_included_outputs():

    pfs = Pfs("tests/testdata/lake.sw")
    df = pfs.get_outputs(section="SPECTRAL_WAVE_MODULE", included_only=True)

    assert df["file_name"][1] == "Wave_parameters.dfsu"
    assert df.shape[0] == 3

    # df.to_csv("outputs.csv")


def test_output_by_id():

    pfs = Pfs("tests/testdata/lake.sw")
    df = pfs.get_outputs(section="SPECTRAL_WAVE_MODULE", included_only=False)
    # .loc refers to output_id irrespective of included or not
    assert df.loc[3]["file_name"] == "Waves_x20km_y20km.dfs0"

    df_inc = pfs.get_outputs(section="SPECTRAL_WAVE_MODULE", included_only=True)
    # .loc refers to output_id irrespective of included or not
    assert df_inc.loc[3]["file_name"] == "Waves_x20km_y20km.dfs0"


def test_encoding():
    Pfs("tests/testdata/OresundHD2D_EnKF10.m21fm")


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_encoding_linux():
    with pytest.raises(ValueError):
        Pfs("tests/testdata/OresundHD2D_EnKF10.m21fm", encoding=None)
