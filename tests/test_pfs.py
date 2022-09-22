import sys
import os
import pytest
import mikeio
import pandas as pd
from datetime import datetime


@pytest.fixture
def d1():
    return dict(
        key1=1,
        lst=[0.3, 0.7],
        SMILE=r"|file\path.dfs|",
        dt=datetime(1979, 2, 3, 3, 5, 0),
        empty=dict(),
    )


@pytest.fixture
def df1():
    d = dict(
        name=["Viken", "Drogden"], position=[[12.5817, 56.128], [12.7113, 55.5364]]
    )
    return pd.DataFrame(d, index=range(1, 3))


def test_pfssection(d1):
    sct = mikeio.PfsSection(d1)
    assert sct.key1 == 1
    assert list(sct.keys()) == ["key1", "lst", "SMILE", "dt", "empty"]
    assert sct["SMILE"] == r"|file\path.dfs|"
    assert len(sct.lst) == 2
    assert sct.dt == datetime(1979, 2, 3, 3, 5, 0)


def test_pfssection_keys_values_items(d1):
    sct = mikeio.PfsSection(d1)
    vals = list(sct.values())
    keys = list(sct.keys())
    j = 0
    for k, v in sct.items():
        assert k == keys[j]
        assert v == vals[j]
        j += 1


def test_pfssection_from_dataframe(df1):
    sct = mikeio.PfsSection.from_dataframe(df1, prefix="MEASUREMENT_")
    assert sct.MEASUREMENT_1.name == "Viken"
    assert sct.MEASUREMENT_2.position == [12.7113, 55.5364]


def test_pfssection_to_dict(d1):
    sct = mikeio.PfsSection(d1)
    d2 = sct.to_dict()
    assert d1 == d2


def test_pfssection_get(d1):
    sct = mikeio.PfsSection(d1)

    v1 = sct.get("key1")
    assert v1 == 1
    assert hasattr(sct, "key1")

    v99 = sct.get("key99")
    assert v99 is None

    v99 = sct.get("key99", "default")
    assert v99 == "default"


def test_pfssection_pop(d1):
    sct = mikeio.PfsSection(d1)

    assert hasattr(sct, "key1")
    v1 = sct.pop("key1")
    assert v1 == 1
    assert not hasattr(sct, "key1")

    v99 = sct.pop("key99", None)
    assert v99 is None


def test_pfssection_del(d1):
    sct = mikeio.PfsSection(d1)

    assert hasattr(sct, "key1")
    del sct["key1"]
    assert not hasattr(sct, "key1")


def test_pfssection_clear(d1):
    sct = mikeio.PfsSection(d1)

    assert hasattr(sct, "key1")
    sct.clear()
    assert not hasattr(sct, "key1")
    assert sct.__dict__ == dict()


def test_pfssection_copy(d1):
    sct1 = mikeio.PfsSection(d1)
    sct2 = sct1  # not copy, only reference

    sct2.key1 = 2
    assert sct1.key1 == 2
    assert sct2.key1 == 2

    sct3 = sct1.copy()
    sct3.key1 = 3
    assert sct3.key1 == 3
    assert sct1.key1 == 2


def test_pfssection_copy_nested(d1):
    sct1 = mikeio.PfsSection(d1)
    sct1["nest"] = mikeio.PfsSection(d1)

    sct3 = sct1.copy()
    sct3.nest.key1 = 2
    assert sct3.nest.key1 == 2
    assert sct1.nest.key1 == 1


def test_pfssection_setitem_update(d1):
    sct = mikeio.PfsSection(d1)
    assert sct.key1 == 1

    sct["key1"] = 2
    assert sct.key1 == 2
    assert sct["key1"] == 2

    sct.key1 = 3
    assert sct.key1 == 3
    assert sct["key1"] == 3


def test_pfssection_setitem_insert(d1):
    sct = mikeio.PfsSection(d1)

    assert not hasattr(sct, "key99")
    sct["key99"] = 99
    assert sct["key99"] == 99
    assert sct.key99 == 99
    assert hasattr(sct, "key99")


def test_pfssection_insert_pfssection(d1):
    sct = mikeio.PfsSection(d1)

    for j in range(10):
        dj = dict(val=j, lst=[0.3, 0.7])
        key = f"FILE_{j+1}"
        sct[key] = mikeio.PfsSection(dj)

    assert sct.FILE_6.val == 5


def test_pfssection_find_replace(d1):
    sct = mikeio.PfsSection(d1)

    for j in range(10):
        dj = dict(val=j, lst=[0.3, 0.7])
        key = f"FILE_{j+1}"
        sct[key] = mikeio.PfsSection(dj)

    assert sct.FILE_6.val == 5
    sct.update_recursive("val", 44)
    assert sct.FILE_6.val == 44


def test_pfssection_find_replace(d1):
    sct = mikeio.PfsSection(d1)

    for j in range(10):
        dj = dict(val=j, lst=[0.3, 0.7])
        key = f"FILE_{j+1}"
        sct[key] = mikeio.PfsSection(dj)

    assert sct.FILE_6.lst == [0.3, 0.7]
    sct.find_replace([0.3, 0.7], [0.11, 0.22, 0.33])
    assert sct.FILE_6.lst == [0.11, 0.22, 0.33]


def test_pfssection_write(d1, tmpdir):
    sct = mikeio.PfsSection(d1)
    pfs = sct.to_Pfs(name="root")
    fn = os.path.join(tmpdir.dirname, "pfssection.pfs")
    pfs.write(fn)

    pfs2 = mikeio.Pfs(fn)
    assert pfs2.data.key1 == sct.key1


def test_basic():

    pfs = mikeio.Pfs("tests/testdata/pfs/simple.pfs")

    data = pfs.data
    # On a pfs file with a single target, the target is implicit,
    #  i.e. BoundaryExtractor in this case

    assert data.z_min == -3000
    assert data.POINT_1.y == 50


def test_ecolab():
    pfs = mikeio.Pfs("tests/testdata/pfs/minimal.ecolab")
    assert pfs.ECO_LAB_SETUP.MISC.DESCRIPTION == "Miscellaneous Description"
    assert pfs.ECO_LAB_SETUP.PROCESSES.PROCESS_1.SCOPE == "WC"

    d1 = pfs.ECO_LAB_SETUP.DERIVED_OUTPUTS.DERIVED_OUTPUT_1
    assert d1.SPATIAL_VARIATION == "HORISONTAL_AND_VERTICAL"


def test_mztoolbox():
    pfs = mikeio.Pfs("tests/testdata/pfs/concat.mzt")
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
    infilename = "tests/testdata/pfs/concat.mzt"
    pfs1 = mikeio.Pfs(infilename)
    outfilename = os.path.join(tmpdir.dirname, "concat_out.mzt")
    pfs1.write(outfilename)
    assert_txt_files_match(infilename, outfilename)
    _ = mikeio.Pfs(outfilename)  # try to parse it also


def test_sw():

    pfs = mikeio.Pfs("tests/testdata/pfs/lake.sw")
    assert pfs.data == pfs.FemEngineSW
    data = pfs.data

    # On a pfs file with a single target, the target is implicit,
    #  i.e. FemEngineSW in this case

    assert data.SPECTRAL_WAVE_MODULE.SPECTRAL.number_of_frequencies == 25
    assert data.SPECTRAL_WAVE_MODULE.SPECTRAL.number_of_directions == 16

    # use shorthand alias SW instead of SPECTRAL_WAVE_MODULE
    assert pfs.SW.SPECTRAL.number_of_frequencies == 25
    assert pfs.SW.WIND.format == 1

    assert data.TIME.number_of_time_steps == 450

    assert data.TIME.start_time.year == 2002
    assert data.TIME.start_time.month == 1


def test_pfssection_to_dataframe():
    pfs = mikeio.Pfs("tests/testdata/pfs/lake.sw")
    df = pfs.SW.OUTPUTS.to_dataframe()
    assert df["file_name"][1] == "Wave_parameters.dfsu"
    assert df.shape[0] == 4


def test_hd_outputs():

    with pytest.warns(match="defined multiple times"):
        pfs = mikeio.Pfs("tests/testdata/pfs/lake.m21fm")
    df = pfs.HD.OUTPUTS.to_dataframe()

    assert df["file_name"][2] == "ts.dfs0"
    assert df.shape[0] == 3


def test_included_outputs():

    pfs = mikeio.Pfs("tests/testdata/pfs/lake.sw")
    df = pfs.SW.OUTPUTS.to_dataframe()
    df = df[df.include == 1]
    # df = pfs.get_outputs(section="SPECTRAL_WAVE_MODULE", included_only=True)

    assert df["file_name"][1] == "Wave_parameters.dfsu"
    assert df.shape[0] == 3

    # df.to_csv("outputs.csv")


def test_output_by_id():

    pfs = mikeio.Pfs("tests/testdata/pfs/lake.sw")
    df = pfs.SW.OUTPUTS.to_dataframe()
    # df = pfs.get_outputs(section="SPECTRAL_WAVE_MODULE", included_only=False)
    # .loc refers to output_id irrespective of included or not
    assert df.loc[3]["file_name"] == "Waves_x20km_y20km.dfs0"

    # df_inc = pfs.get_outputs(section="SPECTRAL_WAVE_MODULE", included_only=True)
    df_inc = df[df.include == 1]
    # .loc refers to output_id irrespective of included or not
    assert df_inc.loc[3]["file_name"] == "Waves_x20km_y20km.dfs0"


def test_encoding():
    with pytest.warns(match="defined multiple times"):
        pfs = mikeio.Pfs("tests/testdata/pfs/OresundHD2D_EnKF10.m21fm")
    assert hasattr(pfs, "DA")


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_encoding_linux():
    with pytest.raises(ValueError):
        mikeio.Pfs("tests/testdata/pfs/OresundHD2D_EnKF10.m21fm", encoding=None)


def test_multiple_identical_roots():
    #    """Test a file created with Mike Zero toolbox containing two similar extraction tasks"""
    pfs = mikeio.read_pfs("tests/testdata/pfs/t1_t0.mzt")
    # assert pfs.data[0].t1_t0.Setup.X == 0
    # assert pfs.data[1].t1_t0.Setup.X == 2
    assert pfs.data[0].Setup.X == 0
    assert pfs.data[1].Setup.X == 2
    assert pfs.t1_t0[0].Setup.X == 0
    assert pfs.t1_t0[1].Setup.X == 2
    assert pfs.names == ["t1_t0", "t1_t0"]
    assert pfs.n_targets == 2


def test_multiple_unique_roots():
    """Test a file created with Mike Zero toolbox containing two similar extraction tasks"""
    pfs = mikeio.read_pfs("tests/testdata/pfs/multiple_unique_root_elements.pfs")
    assert pfs.names == ["FIRST", "MZ_WAVE_SPECTRA_CONVERTER", "SYSTEM"]
    assert pfs.n_targets == 3
    assert not pfs.FIRST.Is_Useful
    assert pfs.MZ_WAVE_SPECTRA_CONVERTER.Setup.Name == "Setup AB"
    assert pfs.SYSTEM.Hi == "there"


def test_multiple_roots_mixed():
    """Test a file created with Mike Zero toolbox containing two similar extraction tasks"""
    pfs = mikeio.read_pfs("tests/testdata/pfs/multiple_root_elements.pfs")
    # assert pfs.target_names == ["t1_t0", "t1_t0"]
    assert pfs.n_targets == 3


def test_non_unique_keywords():
    fn = "tests/testdata/pfs/nonunique.pfs"
    with pytest.warns(match="Keyword z_min defined multiple times"):
        pfs = mikeio.Pfs(fn)

    assert len(pfs.BoundaryExtractor.POINT_1) == 2
    assert isinstance(pfs.BoundaryExtractor.POINT_1[1], mikeio.PfsSection)


def test_illegal_pfs():
    fn = "tests/testdata/pfs/illegal.pfs"
    with pytest.raises(ValueError, match="]]"):
        mikeio.Pfs(fn)


def test_mdf():
    """MDF with multiline polygon"""
    fn = "tests/testdata/pfs/oresund.mdf"

    pfs = mikeio.Pfs(fn)
    assert (
        pfs.data[2].POLYGONS.Data
        == "8 8 Untitled 359236.79224376212 6168403.076453222 1 0 -1 -1 16711680 65535 3 0 1 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 3 38 32 25 8 Untitled 367530.58488032949 6174892.7846391136 0 0 -1 -1 16711680 65535 3 1 700000 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 14 34 25 32 39 1 37 35 31 23 26 17 30 22 24 8 Untitled 358191.86702583247 6164004.5695307152 1 0 -1 -1 16711680 65535 3 0 1 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 2 1 36 8 Untitled 356300.2080261847 6198016.2887355704 1 0 -1 -1 16711680 65535 3 0 1 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 2 9 0 9 Ndr Roese 355957.23455536627 6165986.6140259188 0 0 -1 -1 16711680 65535 3 1 180000 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 6 33 37 36 39 38 34 16 Area of interest 355794.66401566722 6167799.1149176853 0 0 -1 -1 16711680 65535 3 1 50000 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 1 40 8 Untitled 353529.91916129418 6214840.5979535272 0 0 -1 -1 16711680 65535 3 1 700000 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 8 41 8 7 27 4 6 11 12 8 Untitled 351165.00127937191 6173083.0605236143 1 0 -1 -1 16711680 65535 3 0 1 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 1 2 "
    )
