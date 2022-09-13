import sys
import os
import pytest
import mikeio


@pytest.fixture
def d1():
    return dict(key1=1, lst=[0.3, 0.7], SMILE=r"|file\path.dfs|")


def test_pfssection_from_dict(d1):
    sct = mikeio.PfsSection(d1)
    assert sct.key1 == 1
    assert list(sct.keys()) == ["key1", "lst", "SMILE"]
    assert sct["SMILE"] == r"|file\path.dfs|"
    assert len(sct.lst) == 2


def test_pfssection_to_dict(d1):
    sct = mikeio.PfsSection(d1)
    d2 = sct.to_dict()
    assert d1 == d2


def test_basic():

    pfs = mikeio.Pfs("tests/testdata/simple.pfs")

    data = pfs.data
    # On a pfs file with a single target, the target is implicit,
    #  i.e. BoundaryExtractor in this case

    assert data.z_min == -3000
    assert data.POINT_1.y == 50


def test_mztoolbox():
    pfs = mikeio.Pfs("tests/testdata/concat.mzt")
    assert "tide1.dfs" in pfs.data.Setup.File_1.InputFile
    assert "|" in pfs.data.Setup.File_1.InputFile


def assert_files_match(f1, f2):
    with open(f1) as file:
        file1txt = file.read()

    with open(f2) as file:
        file2txt = file.read()

    assert file1txt == file2txt


def assert_txt_files_match(f1, f2, comment="//") -> None:
    """Checks non"""
    with open(f1) as file:
        file1lines = file.read().split("\n")

    with open(f2) as file:
        file2lines = file.read().split("\n")

    for a, b in zip(file1lines, file2lines):
        s1 = a.strip()
        s2 = b.strip()
        if s1 == "" or s1.startswith(comment):
            continue
        if s2 == "" or s2.startswith(comment):
            continue

        assert s1 == s2


def test_read_write(tmpdir):
    infilename = "tests/testdata/concat.mzt"
    pfs1 = mikeio.Pfs(infilename)
    outfilename = os.path.join(tmpdir.dirname, "concat_out.mzt")
    pfs1.write(outfilename)
    assert_txt_files_match(infilename, outfilename)
    _ = mikeio.Pfs(outfilename)  # try to parse it also


def test_sw():

    pfs = mikeio.Pfs("tests/testdata/lake.sw")
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

    pfs = mikeio.Pfs("tests/testdata/lake.sw")
    df = pfs.get_outputs(section="SPECTRAL_WAVE_MODULE")

    assert df["file_name"][1] == "Wave_parameters.dfsu"


# def test_sw_outputs():

#     pfs = mikeio.Pfs("tests/testdata/lake.sw")
#     df = pfs.data.SW.get_outputs()

#     assert df["file_name"][1] == "Wave_parameters.dfsu"
#     assert df.shape[0] == 4

#     df = pfs.data.SW.get_outputs(included_only=True)

#     assert df["file_name"][1] == "Wave_parameters.dfsu"
#     assert df.shape[0] == 3


# def test_hd_outputs():

#     pfs = mikeio.Pfs("tests/testdata/lake.m21fm")
#     df = pfs.data.HD.get_outputs()

#     assert df["file_name"][2] == "ts.dfs0"
#     assert df.shape[0] == 3

#     df = pfs.data.HD.get_outputs(included_only=True)

#     assert df.shape[0] == 2


def test_included_outputs():

    pfs = mikeio.Pfs("tests/testdata/lake.sw")
    df = pfs.get_outputs(section="SPECTRAL_WAVE_MODULE", included_only=True)

    assert df["file_name"][1] == "Wave_parameters.dfsu"
    assert df.shape[0] == 3

    # df.to_csv("outputs.csv")


def test_output_by_id():

    pfs = mikeio.Pfs("tests/testdata/lake.sw")
    df = pfs.get_outputs(section="SPECTRAL_WAVE_MODULE", included_only=False)
    # .loc refers to output_id irrespective of included or not
    assert df.loc[3]["file_name"] == "Waves_x20km_y20km.dfs0"

    df_inc = pfs.get_outputs(section="SPECTRAL_WAVE_MODULE", included_only=True)
    # .loc refers to output_id irrespective of included or not
    assert df_inc.loc[3]["file_name"] == "Waves_x20km_y20km.dfs0"


def test_encoding():
    mikeio.Pfs("tests/testdata/OresundHD2D_EnKF10.m21fm")


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_encoding_linux():
    with pytest.raises(ValueError):
        mikeio.Pfs("tests/testdata/OresundHD2D_EnKF10.m21fm", encoding=None)
