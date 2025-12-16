import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

import mikeio


def assert_txt_files_match(f1, f2, comment="//") -> None:
    """Checks that non-comment lines in two files match exactly.
    Empty lines and lines starting with the comment string are ignored."""
    file1lines = Path(f1).read_text().splitlines()
    file2lines = Path(f2).read_text().splitlines()

    # Filter out comments and empty lines
    content1 = [
        line.strip()
        for line in file1lines
        if line.strip() and not line.strip().startswith(comment)
    ]
    content2 = [
        line.strip()
        for line in file2lines
        if line.strip() and not line.strip().startswith(comment)
    ]

    # Check lengths match after filtering
    if len(content1) != len(content2):
        raise AssertionError(
            f"Files have different number of non-comment lines: {len(content1)} vs {len(content2)}"
        )

    # Compare remaining lines
    for i, (line1, line2) in enumerate(zip(content1, content2)):
        if line1 != line2:
            raise AssertionError(f"Line {i} differs:\n{line1}\nvs\n{line2}")


@pytest.fixture
def d1() -> dict:
    return dict(
        key1=1,
        lst=[0.3, 0.7],
        SMILE=r"|file\path.dfs|",
        dt=datetime(1979, 2, 3, 3, 5, 0),
        empty=dict(),
    )


def test_pfssection(d1) -> None:
    sct = mikeio.PfsSection(d1)
    assert sct.key1 == 1
    assert list(sct.keys()) == ["key1", "lst", "SMILE", "dt", "empty"]
    assert sct["SMILE"] == r"|file\path.dfs|"
    assert len(sct.lst) == 2
    assert sct.dt == datetime(1979, 2, 3, 3, 5, 0)


def test_pfssection_repr(d1) -> None:
    sct = mikeio.PfsSection(d1)
    txt = repr(sct)
    assert len(txt) > 1
    assert "dt = 1979, 2, 3, 3, 5, 0" in txt
    assert "EndSect" in txt


def test_pfs_repr(d1) -> None:
    pfs = mikeio.PfsDocument({"ROOT": d1})
    txt = repr(pfs)
    assert len(txt) > 1
    assert "SMILE = |file" in txt


def test_create_pfsdocument_from_dict(d1) -> None:
    sct = mikeio.PfsDocument({"root": d1})
    assert sct.root.key1 == 1

    with pytest.raises(ValueError, match="all targets must be PfsSections"):
        mikeio.PfsDocument(d1)


def test_pfssection_keys_values_items(d1) -> None:
    sct = mikeio.PfsSection(d1)
    vals = list(sct.values())
    keys = list(sct.keys())
    j = 0
    for k, v in sct.items():
        assert k == keys[j]
        assert v == vals[j]
        j += 1


def test_pfssection_from_dataframe() -> None:
    stations = dict(
        name=["Viken", "Drogden"], position=[[12.5817, 56.128], [12.7113, 55.5364]]
    )
    df = pd.DataFrame(stations, index=range(1, 3))
    sct = mikeio.PfsSection.from_dataframe(df, prefix="MEASUREMENT_")
    assert sct.MEASUREMENT_1.name == "Viken"
    assert sct.MEASUREMENT_2.position == [12.7113, 55.5364]

    with pytest.raises(AttributeError):
        sct.MEASUREMENT_0


def test_pfssection_output_to_and_from_dataframe() -> None:
    pfs = mikeio.PfsDocument("tests/testdata/pfs/lake.sw")
    sct_orig = pfs.SW.OUTPUTS
    df = sct_orig.to_dataframe()
    sct = mikeio.PfsSection.from_dataframe(df, prefix="OUTPUT_")

    # The lines that belong to the [OUTPUTS] is lost in the transformation to a dataframe
    """
       [OUTPUTS]
         Touched = 1
         MzSEPfsListItemCount = 4
         number_of_outputs = 4
         [OUTPUT_1]
            ...
         [OUTPUT_2]
            ...
    """
    orig_lines = str(sct_orig).splitlines()
    new_lines = str(sct).splitlines()

    assert "number_of_outputs" in orig_lines[2]

    n_new_lines = len(new_lines)
    n_orig_lines = len(orig_lines)
    n_missing_lines = n_orig_lines - n_new_lines

    assert orig_lines[n_missing_lines:] == new_lines


def test_pfssection_to_dict(d1) -> None:
    sct = mikeio.PfsSection(d1)
    d2 = sct.to_dict()
    assert d1 == d2


def test_pfssection_get(d1) -> None:
    sct = mikeio.PfsSection(d1)

    v1 = sct.get("key1")
    assert v1 == 1
    assert hasattr(sct, "key1")

    v99 = sct.get("key99")
    assert v99 is None

    v99 = sct.get("key99", "default")
    assert v99 == "default"


def test_pfssection_pop(d1) -> None:
    sct = mikeio.PfsSection(d1)

    assert hasattr(sct, "key1")
    v1 = sct.pop("key1")
    assert v1 == 1
    assert not hasattr(sct, "key1")

    v99 = sct.pop("key99", None)
    assert v99 is None


def test_pfssection_del(d1) -> None:
    sct = mikeio.PfsSection(d1)

    assert hasattr(sct, "key1")
    del sct["key1"]
    assert not hasattr(sct, "key1")


def test_pfssection_clear(d1) -> None:
    sct = mikeio.PfsSection(d1)

    assert hasattr(sct, "key1")
    sct.clear()
    assert not hasattr(sct, "key1")
    assert sct.__dict__ == dict()


def test_pfssection_copy(d1) -> None:
    sct1 = mikeio.PfsSection(d1)
    sct2 = sct1  # not copy, only reference

    sct2.key1 = 2
    assert sct1.key1 == 2
    assert sct2.key1 == 2

    sct3 = sct1.copy()
    sct3.key1 = 3
    assert sct3.key1 == 3
    assert sct1.key1 == 2


def test_pfssection_len(d1) -> None:
    sct = mikeio.PfsSection(d1)
    assert len(sct) == 5


def test_pfssection_contains(d1) -> None:
    sct = mikeio.PfsSection(d1)
    assert "key1" in sct
    assert "key2" not in sct


def test_pfssection_copy_nested(d1) -> None:
    sct1 = mikeio.PfsSection(d1)
    sct1["nest"] = mikeio.PfsSection(d1)

    sct3 = sct1.copy()
    sct3.nest.key1 = 2
    assert sct3.nest.key1 == 2
    assert sct1.nest.key1 == 1


def test_pfssection_setitem_update(d1) -> None:
    sct = mikeio.PfsSection(d1)
    assert sct.key1 == 1

    sct["key1"] = 2
    assert sct.key1 == 2
    assert sct["key1"] == 2

    sct.key1 = 3
    assert sct.key1 == 3
    assert sct["key1"] == 3


def test_pfssection_setitem_insert(d1) -> None:
    sct = mikeio.PfsSection(d1)

    assert not hasattr(sct, "key99")
    sct["key99"] = 99
    assert sct["key99"] == 99
    assert sct.key99 == 99
    assert hasattr(sct, "key99")


def test_pfssection_insert_pfssection(d1) -> None:
    sct = mikeio.PfsSection(d1)

    for j in range(10):
        dj = dict(val=j, lst=[0.3, 0.7])
        key = f"FILE_{j + 1}"
        sct[key] = mikeio.PfsSection(dj)

    assert sct.FILE_6.val == 5


def test_pfssection_find_replace(d1) -> None:
    sct = mikeio.PfsSection(d1)

    for j in range(10):
        dj = dict(val=j, lst=[0.3, 0.7])
        key = f"FILE_{j + 1}"
        sct[key] = mikeio.PfsSection(dj)

    assert sct.FILE_6.lst == [0.3, 0.7]
    sct.find_replace([0.3, 0.7], [0.11, 0.22, 0.33])
    assert sct.FILE_6.lst == [0.11, 0.22, 0.33]


def test_pfssection_write(d1, tmp_path: Path) -> None:
    sct = mikeio.PfsSection(d1)
    pfs = mikeio.PfsDocument({"root": sct})
    fn = tmp_path / "pfssection.pfs"
    pfs.write(fn)

    pfs2 = mikeio.PfsDocument(fn)
    assert pfs2.root.key1 == sct.key1


def test_str_is_scientific_float(d1) -> None:
    sect = mikeio.PfsSection(d1)  # dummy
    func = sect._str_is_scientific_float
    assert func("-1.0e2")
    assert func("1E-4")
    assert func("-0.123213e-23")
    assert not func("-0.1E+0.5")
    assert not func("E12")
    assert not func("E-4")
    assert not func("-1.0e2e")
    assert not func("e-1.0e2")


def test_basic() -> None:
    pfs = mikeio.PfsDocument("tests/testdata/pfs/simple.pfs")

    data = pfs.targets[0]
    assert pfs.targets[0] == pfs.BoundaryExtractor

    assert data.z_min == -3000
    assert data.POINT_1.y == 50


def test_pfsdocument_copy() -> None:
    pfs = mikeio.PfsDocument("tests/testdata/pfs/simple.pfs")
    pfs2 = pfs.copy()
    data = pfs.targets[0]
    data.z_min = -4000

    assert pfs2.BoundaryExtractor.z_min == -3000


def test_ecolab() -> None:
    pfs = mikeio.PfsDocument("tests/testdata/pfs/minimal.ecolab")
    assert pfs.ECO_LAB_SETUP.MISC.DESCRIPTION == "Miscellaneous Description"
    assert pfs.ECO_LAB_SETUP.PROCESSES.PROCESS_1.SCOPE == "WC"

    d1 = pfs.ECO_LAB_SETUP.DERIVED_OUTPUTS.DERIVED_OUTPUT_1
    assert d1.SPATIAL_VARIATION == "HORISONTAL_AND_VERTICAL"


def test_mztoolbox() -> None:
    pfs = mikeio.PfsDocument("tests/testdata/pfs/concat.mzt")
    assert "tide1.dfs" in pfs.txconc.Setup.File_1.InputFile
    assert "|" in pfs.txconc.Setup.File_1.InputFile


def test_read_write(tmp_path: Path) -> None:
    infilename = "tests/testdata/pfs/concat.mzt"
    pfs1 = mikeio.PfsDocument(infilename)
    outfilename = tmp_path / "concat_out.mzt"
    pfs1.write(outfilename)
    assert_txt_files_match(infilename, outfilename)
    _ = mikeio.PfsDocument(outfilename)  # try to parse it also


def test_read_write_she(tmp_path: Path) -> None:
    infilename = "tests/testdata/pfs/Karup_basic.she"
    pfs1 = mikeio.PfsDocument(infilename, unique_keywords=False)
    outfilename = tmp_path / "Karup_basic_out.she"
    pfs1.write(outfilename)
    pfs2 = mikeio.PfsDocument(outfilename)
    assert pfs1.MIKESHE_FLOWMODEL == pfs2.MIKESHE_FLOWMODEL


def test_read_write_she2(tmp_path: Path) -> None:
    infilename = "tests/testdata/pfs/Karup_mini.she"
    with pytest.warns(match="contains a single quote character"):
        pfs1 = mikeio.PfsDocument(infilename)

    outfilename = tmp_path / "Karup_mini_out.she"
    pfs1.write(outfilename)

    with pytest.warns(match="contains a single quote character"):
        pfs2 = mikeio.PfsDocument(outfilename)
    assert pfs1.MIKESHE_FLOWMODEL == pfs2.MIKESHE_FLOWMODEL


def test_read_write_filenames(tmp_path: Path) -> None:
    infilename = "tests/testdata/pfs/filenames.pfs"
    pfs1 = mikeio.PfsDocument(infilename)
    outfilename = tmp_path / "filenames_out.pfs"
    pfs1.write(outfilename)
    assert_txt_files_match(infilename, outfilename)
    _ = mikeio.PfsDocument(outfilename)  # try to parse it also


def test_read_write_filenames_modified(tmp_path: Path) -> None:
    infilename = "tests/testdata/pfs/filenames.pfs"
    pfs1 = mikeio.PfsDocument(infilename)
    pfs1.FILE_NAMES.file5 = r"..\..\newfile5.dfs0"
    pfs1.FILE_NAMES.file6 = "|../newfile6.dfs0|"
    outfilename = tmp_path / "filenames_out.pfs"
    pfs1.write(outfilename)

    pfs2 = mikeio.PfsDocument(outfilename)
    d1 = pfs1.to_dict()
    d2 = pfs2.to_dict()
    assert d1 == d2


def test_sw() -> None:
    pfs = mikeio.PfsDocument("tests/testdata/pfs/lake.sw")
    assert pfs.targets[0] == pfs.FemEngineSW
    root = pfs.targets[0]

    assert root.SPECTRAL_WAVE_MODULE.SPECTRAL.number_of_frequencies == 25
    assert root.SPECTRAL_WAVE_MODULE.SPECTRAL.number_of_directions == 16

    # use shorthand alias SW instead of SPECTRAL_WAVE_MODULE
    assert pfs.SW.SPECTRAL.number_of_frequencies == 25
    assert pfs.SW.WIND.format == 1

    assert root.TIME.number_of_time_steps == 450

    # we're no longer parsing datetimes in Pfs module
    start_time = datetime(*root.TIME.start_time)
    assert start_time.year == 2002
    assert start_time.month == 1


def test_pfssection_to_dataframe() -> None:
    pfs = mikeio.PfsDocument("tests/testdata/pfs/lake.sw")
    df = pfs.SW.OUTPUTS.to_dataframe()
    assert df["file_name"][1] == "Wave_parameters.dfsu"
    assert df.shape[0] == 4


def test_hd_outputs() -> None:
    with pytest.warns(match="defined multiple times"):
        pfs = mikeio.PfsDocument("tests/testdata/pfs/lake.m21fm", unique_keywords=True)
    df = pfs.HD.OUTPUTS.to_dataframe()

    assert df["file_name"][2] == "ts.dfs0"
    assert df.shape[0] == 3


def test_included_outputs() -> None:
    pfs = mikeio.PfsDocument("tests/testdata/pfs/lake.sw")
    df = pfs.SW.OUTPUTS.to_dataframe(prefix="OUTPUT_")
    df = df[df.include == 1]

    assert df["file_name"][1] == "Wave_parameters.dfsu"
    assert df.shape[0] == 3


def test_output_by_id() -> None:
    pfs = mikeio.PfsDocument("tests/testdata/pfs/lake.sw")
    df = pfs.SW.OUTPUTS.to_dataframe()
    # .loc refers to output_id irrespective of included or not
    assert df.loc[3]["file_name"] == "Waves_x20km_y20km.dfs0"

    df_inc = df[df.include == 1]
    # .loc refers to output_id irrespective of included or not
    assert df_inc.loc[3]["file_name"] == "Waves_x20km_y20km.dfs0"


def test_encoding() -> None:
    with pytest.warns(match="defined multiple times"):
        pfs = mikeio.PfsDocument(
            "tests/testdata/pfs/OresundHD2D_EnKF10.m21fm", unique_keywords=True
        )
    assert len(pfs.keys()) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_encoding_linux() -> None:
    with pytest.raises(ValueError):
        mikeio.PfsDocument("tests/testdata/pfs/OresundHD2D_EnKF10.m21fm", encoding=None)  # type: ignore


def test_multiple_identical_roots() -> None:
    """Test a file created with Mike Zero toolbox containing two similar extraction tasks"""
    pfs = mikeio.read_pfs("tests/testdata/pfs/t1_t0.mzt")
    assert pfs.targets[0].Setup.X == 0
    assert pfs.targets[1].Setup.X == 2
    assert pfs.t1_t0[0].Setup.X == 0
    assert pfs.t1_t0[1].Setup.X == 2
    assert pfs.names == ["t1_t0", "t1_t0"]
    assert pfs.n_targets == 2
    assert not pfs.is_unique


def test_multiple_roots_from_text() -> None:
    text = """
[t1_t0]
   [Setup]
      Name = 'Extract this'
   EndSect  // Setup
EndSect  // t1_t0

[t1_t0]
   [Setup]
      Name = 'Extract that'
   EndSect  // Setup
EndSect  // t1_t0
    """
    pfs = mikeio.PfsDocument.from_text(text)
    assert pfs.targets[0].Setup.Name == "Extract this"
    assert pfs.targets[1].Setup.Name == "Extract that"


def test_multiple_roots_from_data() -> None:
    data = [
        mikeio.PfsSection({"ATOOL": {"file_name": r"|path\file1.dfs0|"}}),
        mikeio.PfsSection({"ATOOL": {"file_name": r"|path\file2.dfs0|"}}),
    ]

    pfs = mikeio.PfsDocument(data)
    assert pfs.names[0] == "ATOOL"
    assert pfs.names[1] == "ATOOL"
    assert pfs.targets[0].file_name == r"|path\file1.dfs0|"
    assert pfs.targets[1].file_name == r"|path\file2.dfs0|"


def test_multiple_unique_roots() -> None:
    """Test a file created with Mike Zero toolbox containing two similar extraction tasks"""
    pfs = mikeio.read_pfs("tests/testdata/pfs/multiple_unique_root_elements.pfs")
    assert pfs.names == ["FIRST", "MZ_WAVE_SPECTRA_CONVERTER", "SYSTEM"]
    assert pfs.n_targets == 3
    assert not pfs.FIRST.Is_Useful
    assert pfs.MZ_WAVE_SPECTRA_CONVERTER.Setup.Name == "Setup AB"
    assert pfs.SYSTEM.Hi == "there"


def test_multiple_roots_mixed() -> None:
    """Test a file created with Mike Zero toolbox containing two similar extraction tasks"""
    pfs = mikeio.read_pfs("tests/testdata/pfs/multiple_root_elements.pfs")
    assert list(pfs.keys()) == [
        "MZ_WAVE_SPECTRA_CONVERTER",
        "SYSTEM",
    ]
    assert pfs.names == [
        "MZ_WAVE_SPECTRA_CONVERTER",
        "MZ_WAVE_SPECTRA_CONVERTER",
        "SYSTEM",
    ]
    assert pfs.n_targets == 3


def test_non_unique_keywords() -> None:
    fn = "tests/testdata/pfs/nonunique.pfs"
    with pytest.warns(match="Keyword z_min defined multiple times"):
        pfs = mikeio.PfsDocument(fn, unique_keywords=True)

    assert len(pfs.BoundaryExtractor.POINT_1) == 2
    assert isinstance(pfs.BoundaryExtractor.POINT_1[1], mikeio.PfsSection)

    # first value will be kept (like MIKE FM)
    assert pfs.BoundaryExtractor.z_min == -3000


def test_non_unique_keywords_allowed() -> None:
    fn = "tests/testdata/pfs/nonunique.pfs"
    pfs = mikeio.PfsDocument(fn, unique_keywords=False)

    assert len(pfs.BoundaryExtractor.POINT_1) == 2
    assert isinstance(pfs.BoundaryExtractor.POINT_1[1], mikeio.PfsSection)

    assert len(pfs.BoundaryExtractor.z_min) == 3
    assert pfs.BoundaryExtractor.z_min == [-3000, 9, 19]


def test_non_unique_keywords_read_write(tmp_path: Path) -> None:
    fn1 = "tests/testdata/pfs/nonunique.pfs"
    pfs1 = mikeio.PfsDocument(fn1, unique_keywords=False)

    fn2 = str(tmp_path / "nonunique_out.pfs")
    pfs1.write(fn2)

    pfs2 = mikeio.PfsDocument(fn2, unique_keywords=False)

    d1 = pfs1.BoundaryExtractor.to_dict()
    d2 = pfs2.BoundaryExtractor.to_dict()
    assert d1 == d2


def test_illegal_pfs() -> None:
    fn = "tests/testdata/pfs/illegal.pfs"
    with pytest.raises(ValueError, match="]]"):
        mikeio.PfsDocument(fn)


def test_mdf() -> None:
    """MDF with multiline polygon"""
    fn = "tests/testdata/pfs/oresund.mdf"

    pfs = mikeio.PfsDocument(fn)
    assert (
        pfs.targets[2].POLYGONS.Data
        == "8 8 Untitled 359236.79224376212 6168403.076453222 1 0 -1 -1 16711680 65535 3 0 1 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 3 38 32 25 8 Untitled 367530.58488032949 6174892.7846391136 0 0 -1 -1 16711680 65535 3 1 700000 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 14 34 25 32 39 1 37 35 31 23 26 17 30 22 24 8 Untitled 358191.86702583247 6164004.5695307152 1 0 -1 -1 16711680 65535 3 0 1 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 2 1 36 8 Untitled 356300.2080261847 6198016.2887355704 1 0 -1 -1 16711680 65535 3 0 1 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 2 9 0 9 Ndr Roese 355957.23455536627 6165986.6140259188 0 0 -1 -1 16711680 65535 3 1 180000 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 6 33 37 36 39 38 34 16 Area of interest 355794.66401566722 6167799.1149176853 0 0 -1 -1 16711680 65535 3 1 50000 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 1 40 8 Untitled 353529.91916129418 6214840.5979535272 0 0 -1 -1 16711680 65535 3 1 700000 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 8 41 8 7 27 4 6 11 12 8 Untitled 351165.00127937191 6173083.0605236143 1 0 -1 -1 16711680 65535 3 0 1 0 1 1000 1000 0 0 0 0 0 1 1000 2 2 0 10 1 2 "
    )


def test_read_in_memory_string() -> None:
    text = """
[ENGINE]
  option = foo,bar
EndSect // ENGINE
"""
    pfs = mikeio.PfsDocument.from_text(text)

    assert pfs.ENGINE.option == ["foo", "bar"]


def test_read_mixed_array() -> None:
    text = """
[ENGINE]
  advanced= false
  fill_list = false, 'TEST'
EndSect // ENGINE
"""
    pfs = mikeio.PfsDocument.from_text(text)

    assert pfs.ENGINE.advanced is False
    assert isinstance(pfs.ENGINE.fill_list, (list, tuple))
    assert len(pfs.ENGINE.fill_list) == 2
    assert pfs.ENGINE.fill_list[0] is False
    assert pfs.ENGINE.fill_list[1] == "TEST"


def test_read_mixed_array2() -> None:
    text = """
[ENGINE]
  fill_list = 'dsd', 0, 0.0, false
EndSect // ENGINE
"""
    pfs = mikeio.PfsDocument.from_text(text)
    assert isinstance(pfs.ENGINE.fill_list, (list, tuple))
    assert len(pfs.ENGINE.fill_list) == 4
    assert pfs.ENGINE.fill_list[0] == "dsd"
    assert pfs.ENGINE.fill_list[1] == 0
    assert pfs.ENGINE.fill_list[2] == 0.0
    assert pfs.ENGINE.fill_list[3] is False


def test_read_mixed_array3() -> None:
    text = """
[ENGINE]
  fill_list = 'dsd', 0, 0.0, "str2", false, 'str3'
EndSect // ENGINE
"""
    pfs = mikeio.PfsDocument.from_text(text)
    assert isinstance(pfs.ENGINE.fill_list, (list, tuple))
    assert len(pfs.ENGINE.fill_list) == 6
    assert pfs.ENGINE.fill_list[0] == "dsd"
    assert pfs.ENGINE.fill_list[1] == 0
    assert pfs.ENGINE.fill_list[2] == 0.0
    assert pfs.ENGINE.fill_list[3] == "str2"
    assert pfs.ENGINE.fill_list[4] is False
    assert pfs.ENGINE.fill_list[5] == "str3"


def test_read_array() -> None:
    text = """
[ENGINE]
  fill_list = 1, 2
EndSect // ENGINE
"""
    pfs = mikeio.PfsDocument.from_text(text)

    assert isinstance(pfs.ENGINE.fill_list, (list, tuple))
    assert len(pfs.ENGINE.fill_list) == 2
    assert pfs.ENGINE.fill_list[0] == 1
    assert pfs.ENGINE.fill_list[1] == 2


def test_empty(tmp_path: Path) -> None:
    text = """
[ENGINE]
  A = 
  [B]
  EndSect // B  
EndSect // ENGINE
"""
    pfs = mikeio.PfsDocument.from_text(text)

    assert isinstance(pfs.ENGINE.A, list)
    assert len(pfs.ENGINE.A) == 0
    assert isinstance(pfs.ENGINE.B, mikeio.PfsSection)
    assert len(pfs.ENGINE.B) == 0

    outfile = tmp_path / "empty.pfs"
    pfs.write(outfile)

    outlines = Path(outfile).read_text().splitlines()

    assert outlines[5].strip() == "A ="
    assert outlines[6].strip() == "[B]"
    assert outlines[7].strip() == "EndSect  // B"


def test_difficult_chars_in_str(tmp_path: Path) -> None:
    text = r"""
[ENGINE]
  A = 'str,s/d\sd.dfs0'
  B = "str,sd'sd.dfs0"
  C = |sd's\d.dfs0|
  D = |str'd.dfs0|
  E = |str,s'+-s_d.dfs0|
EndSect // ENGINE
"""
    with pytest.warns(match="contains a single quote character"):
        pfs = mikeio.PfsDocument.from_text(text)

    assert isinstance(pfs.ENGINE.A, str)
    assert pfs.ENGINE.A == r"str,s/d\sd.dfs0"

    # NOTE: B will appear wrong as a list with one item
    assert isinstance(pfs.ENGINE.B[0], str)
    assert pfs.ENGINE.B[0] == "str,sd'sd.dfs0"
    assert isinstance(pfs.ENGINE.C, str)
    assert pfs.ENGINE.C == "|sd\U0001f600s\\d.dfs0|"
    assert isinstance(pfs.ENGINE.D, str)
    assert pfs.ENGINE.E == "|str,s\U0001f600+-s_d.dfs0|"

    outfile = tmp_path / "difficult_chars_in_str.pfs"
    pfs.write(outfile)

    outlines = Path(outfile).read_text().splitlines()

    assert outlines[5].strip() == "A = 'str,s/d\\sd.dfs0'"
    assert outlines[6].strip() == "B = 'str,sd'sd.dfs0'"
    assert outlines[7].strip() == "C = |sd's\\d.dfs0|"
    assert outlines[8].strip() == "D = |str'd.dfs0|"
    assert outlines[9].strip() == "E = |str,s'+-s_d.dfs0|"


def test_difficult_chars_in_str2(tmp_path: Path) -> None:
    text = r"""
[ENGINE]
   A = 'str,s/d\sd.dfs0'
   B = "str,sd'sd.dfs0"
   C = |str'd.dfs0|
   D = |str,s'+-s_d.dfs0|
EndSect // ENGINE"""

    with pytest.warns(match="contains a single quote character"):
        pfs = mikeio.PfsDocument.from_text(text)

    outfile = tmp_path / "difficult_chars_in_str2.pfs"
    pfs.write(outfile)

    outlines = Path(outfile).read_text().splitlines()

    assert outlines[5].strip() == "A = 'str,s/d\\sd.dfs0'"
    assert outlines[6].strip() == "B = 'str,sd'sd.dfs0'"
    assert outlines[7].strip() == "C = |str'd.dfs0|"
    assert outlines[8].strip() == "D = |str,s'+-s_d.dfs0|"


def test_end_of_stream() -> None:
    text = """
[Results]
  hd = ||, 'm11EcoRes.res11', 1, 0
  ad = ||, '', 1, 0
  st = ||, '', 1, 0
  rr = ||, '', 1, 0
EndSect  // Results
"""
    mikeio.PfsDocument.from_text(text)


def test_read_string_array() -> None:
    text = """
[ENGINE]
  fill_list = 'foo', 'bar', 'baz'
EndSect // ENGINE
"""
    pfs = mikeio.PfsDocument.from_text(text)

    assert isinstance(pfs.ENGINE.fill_list, (list, tuple))
    assert len(pfs.ENGINE.fill_list) == 3
    assert pfs.ENGINE.fill_list[0] == "foo"
    assert pfs.ENGINE.fill_list[1] == "bar"
    assert pfs.ENGINE.fill_list[2] == "baz"


def test_read_write_list_list(tmp_path: Path) -> None:
    text = """
[ENGINE]
  RGB_Color_Value = 128, 0, 128
  RGB_Color_Value = 85, 0, 171
EndSect // ENGINE
"""
    pfs = mikeio.PfsDocument(StringIO(text), unique_keywords=False)
    assert len(pfs.ENGINE.RGB_Color_Value) == 2
    assert len(pfs.ENGINE.RGB_Color_Value[0]) == 3

    outfile = tmp_path / "mini.pal"

    pfs.write(outfile)

    outlines = Path(outfile).read_text().splitlines()

    n_rgb_out = len([line for line in outlines if "RGB_Color_Value" in line])
    assert n_rgb_out == 2


def test_pfs_repr_contains_name_of_target() -> None:
    text = """
[ENGINE]
  RGB_Color_Value = 128, 0, 128
  RGB_Color_Value = 85, 0, 171
EndSect // ENGINE
"""
    pfs = mikeio.PfsDocument(StringIO(text), unique_keywords=False)
    text = repr(pfs)

    # TODO should we be more specific i.e. [ENGINE] ?
    assert "ENGINE" in text


def test_double_single_quotes_in_string(tmp_path: Path) -> None:
    text = """
[DERIVED_VARIABLE_106]
            name = 'alfa_PC_T'
            type = 0
            dimension = 3
            description = 'alfa_PC_T, ''light'' adjusted alfa_PC, ugC/gC*m2/uE'
            EUM_type = 999
            EUM_unit = 0
            unit = 'ugC/gC*m2/uE'
            ID = 597
EndSect  // DERIVED_VARIABLE_106
"""

    pfs = mikeio.PfsDocument.from_text(text)
    assert (
        pfs.DERIVED_VARIABLE_106.description
        == 'alfa_PC_T, "light" adjusted alfa_PC, ugC/gC*m2/uE'
    )

    filename = tmp_path / "quotes.pfs"

    pfs.write(filename)

    with open(filename) as f:
        for line in f:
            if "description" in line:
                assert (
                    line.strip()
                    == "description = 'alfa_PC_T, ''light'' adjusted alfa_PC, ugC/gC*m2/uE'"
                )


def test_str_in_str_projection(tmp_path: Path) -> None:
    text = """
   [ROOT]
      Proj = 'PROJCS["ETRS_1989",GEOGCS["GCS_1989",DATUM["D_ETRS_1"]]]'
   EndSect  // ROOT 
"""

    pfs = mikeio.PfsDocument.from_text(text)
    assert isinstance(pfs.ROOT.Proj, str)
    assert pfs.ROOT.Proj[8] == "E"

    filename = tmp_path / "str_in_str.pfs"
    pfs.write(filename)

    with open(filename) as f:
        for line in f:
            if "Proj" in line:
                assert (
                    line.strip()
                    == 'Proj = \'PROJCS["ETRS_1989",GEOGCS["GCS_1989",DATUM["D_ETRS_1"]]]\''
                )


def test_number_in_str(tmp_path: Path) -> None:
    text = """
   [ROOT]
      ID1 = '1'
      ID2 = "1"
      Number = 1
   EndSect  // ROOT 
"""

    pfs = mikeio.PfsDocument.from_text(text)
    assert isinstance(pfs.ROOT.ID1, str)
    assert isinstance(pfs.ROOT.ID2, str)
    assert not isinstance(pfs.ROOT.Number, str)

    filename = tmp_path / "number_in_str.pfs"
    pfs.write(filename)

    with open(filename) as f:
        for line in f:
            if "ID1" in line:
                assert line.strip() == "ID1 = '1'"
            if "ID2" in line:
                assert line.strip() == "ID2 = '1'"


def test_floatlike_strings(tmp_path: Path) -> None:
    text = """
    [WELLNO_424]
      ID_A = '1E-3'
      ID_B = '1-E'
      ID_C = '1-E3'
    EndSect  // WELLNO_424
"""
    pfs = mikeio.PfsDocument.from_text(text)
    # assert pfs.WELLNO_424.ID_A == "1E-3"  # not possible to distinguish
    assert pfs.WELLNO_424.ID_B == "1-E"
    assert pfs.WELLNO_424.ID_C == "1-E3"

    filename = tmp_path / "float_like_strings.pfs"
    pfs.write(filename)
    pfs = mikeio.read_pfs(filename)
    # assert pfs.WELLNO_424.ID_A == "1E-3"  # not possible to distinguish
    assert pfs.WELLNO_424.ID_B == "1-E"
    assert pfs.WELLNO_424.ID_C == "1-E3"


def test_nested_quotes(tmp_path: Path) -> None:
    text = """
  [Weir_0]
    Properties = '<CLOB:"1495_weir",0,0,false,0.0,0.0,0.0,0.0,1,0,0.5,1.0,1.0,0.5,1.0,1.0,0,0.0,0.0,"00000000-0000-0000-0000-000000000000",15.24018,5.181663,0.0,"","">'
  EndSect  // Weir_0
"""
    pfs = mikeio.PfsDocument.from_text(text)
    assert (
        pfs.Weir_0.Properties
        == '<CLOB:"1495_weir",0,0,false,0.0,0.0,0.0,0.0,1,0,0.5,1.0,1.0,0.5,1.0,1.0,0,0.0,0.0,"00000000-0000-0000-0000-000000000000",15.24018,5.181663,0.0,"","">'
    )

    filename = tmp_path / "nested_quotes.pfs"
    pfs.write(filename)

    with open(filename) as f:
        for line in f:
            if "Properties" in line:
                assert (
                    line.strip()
                    == 'Properties = \'<CLOB:"1495_weir",0,0,false,0.0,0.0,0.0,0.0,1,0,0.5,1.0,1.0,0.5,1.0,1.0,0,0.0,0.0,"00000000-0000-0000-0000-000000000000",15.24018,5.181663,0.0,"","">\''
                )


def test_filename_in_list(tmp_path: Path) -> None:
    text = r"""
   [EcolabTemplateSpecification]
      TemplateFile_A = |.\Test1_OLSZ_OL_WQsetups.ecolab|
      TemplateFile_OL = |.\Test1_OLSZ_OL_WQsetups.ecolab|, 2019, 4, 25, 14, 51, 35
      Method_OL = 0
   EndSect  // EcolabTemplateSpecification 
"""
    pfs = mikeio.PfsDocument.from_text(text)
    assert len(pfs.EcolabTemplateSpecification.TemplateFile_OL) == 7
    assert (
        pfs.EcolabTemplateSpecification.TemplateFile_OL[0]
        == r"|.\Test1_OLSZ_OL_WQsetups.ecolab|"
    )

    filename = tmp_path / "filename_in_list.pfs"
    pfs.write(filename)

    with open(filename) as f:
        for line in f:
            if "TemplateFile_OL" in line:
                assert (
                    line.strip()
                    == r"TemplateFile_OL = |.\Test1_OLSZ_OL_WQsetups.ecolab|, 2019, 4, 25, 14, 51, 35"
                )


def test_multiple_empty_strings_in_list(tmp_path: Path) -> None:
    text = """
   [Engine]
      A = '', '', '', ''
      B = '', '', ||, ||, "", ""
   EndSect  // Engine 
"""

    pfs = mikeio.PfsDocument.from_text(text)
    assert len(pfs.Engine.A) == 4
    assert pfs.Engine.A[0] == ""
    assert pfs.Engine.A[-1] == ""
    assert len(pfs.Engine.B) == 6
    assert pfs.Engine.B[0] == ""
    assert pfs.Engine.B[2] == "||"
    assert pfs.Engine.B[-1] == ""

    filename = tmp_path / "multiple_empty_strings_in_list.pfs"
    pfs.write(filename)

    with open(filename) as f:
        for line in f:
            if "A =" in line:
                assert line.strip() == "A = '', '', '', ''"
            if "B =" in line:
                assert line.strip() == "B = '', '', ||, ||, '', ''"


def test_vertical_lines_in_list(tmp_path: Path) -> None:
    text = """
   [EcolabTemplateSpecification]
      TemplateFile_OL = ||, -1, -1, ||, -1, -1, ||
      Method_OL = 0
   EndSect  // EcolabTemplateSpecification 
"""

    pfs = mikeio.PfsDocument.from_text(text)
    assert len(pfs.EcolabTemplateSpecification.TemplateFile_OL) == 7
    assert pfs.EcolabTemplateSpecification.TemplateFile_OL[0] == "||"
    assert pfs.EcolabTemplateSpecification.TemplateFile_OL[3] == "||"
    assert pfs.EcolabTemplateSpecification.TemplateFile_OL[-1] == "||"

    filename = tmp_path / "vertical_lines_in_list.pfs"
    pfs.write(filename)

    with open(filename) as f:
        for line in f:
            if "TemplateFile_OL" in line:
                assert line.strip() == "TemplateFile_OL = ||, -1, -1, ||, -1, -1, ||"


def test_nonunique_mixed_keywords_sections1(tmp_path: Path) -> None:
    text = """
   [ROOT]
      A = '1'
      A = 0
      [A]
         B = 0
         B = 2
      EndSect  // A
      A = 3
   EndSect  // ROOT 
"""

    pfs = mikeio.PfsDocument.from_text(text)
    assert len(pfs.ROOT.A) == 4
    assert isinstance(pfs.ROOT.A[2], mikeio.PfsSection)
    assert len(pfs.ROOT.A[2].B) == 2
    assert pfs.ROOT.A[2].B[0] == 0
    assert pfs.ROOT.A[-1] == 3

    filename = tmp_path / "nonunique_mixed_keywords_sections.pfs"
    pfs.write(filename)

    outlines = Path(filename).read_text().splitlines()

    assert outlines[5].strip() == "A = '1'"
    assert outlines[6].strip() == "A = 0"
    assert outlines[7].strip() == "[A]"
    assert outlines[8].strip() == "B = 0"
    assert outlines[9].strip() == "B = 2"


def test_nonunique_mixed_keywords_sections2(tmp_path: Path) -> None:
    text = """
   [ROOT]
      [A]
         B = 0
         [B]
            C = 4.5
        EndSect  // B
      EndSect  // A
      A = 0
      [A]
         B = 0
      EndSect  // A
      A = 3
   EndSect  // ROOT 
"""

    pfs = mikeio.PfsDocument.from_text(text)
    assert len(pfs.ROOT.A) == 4
    assert isinstance(pfs.ROOT.A[0], mikeio.PfsSection)
    assert isinstance(pfs.ROOT.A[0].B[1], mikeio.PfsSection)
    assert isinstance(pfs.ROOT.A[2], mikeio.PfsSection)
    assert pfs.ROOT.A[2].B == 0
    assert pfs.ROOT.A[-1] == 3

    filename = tmp_path / "nonunique_mixed_keywords_sections.pfs"
    pfs.write(filename)


def test_parse_mike_she_pfs() -> None:
    pfs = mikeio.PfsDocument("tests/testdata/pfs/Karup_basic.she")

    assert pfs.n_targets == 2
    assert (
        pfs.MIKESHE_FLOWMODEL.SimSpec.ModelComp.River == 1
    )  # TODO Is this sensible to check?


def test_read_write_grid_editor_color_palette(tmp_path: Path) -> None:
    infile = "tests/testdata/pfs/grid1.gsf"
    outfile = tmp_path / "grid.gsf"
    pfs = mikeio.PfsDocument(infile)
    pal = pfs.GRID_EDITOR.GRID_EDIT_VIEW.MIKEZero_Palette_Definition
    assert len(pal.RGB_Color_Value) == 16

    pfs.write(outfile)

    with open(outfile) as f:
        outlines = f.readlines()

    n_rgb_out = len([line for line in outlines if "RGB_Color_Value" in line])
    assert n_rgb_out == 16


@pytest.fixture
def pfs_ABC_text() -> str:
    text = """
   [ROOT]
      [A1]
         int_1 = 0
         [B]
            float_1 = 4.5
        EndSect  // B
      EndSect  // A1
      str_1 = '0'
      [A2]
         int_2 = 0
      EndSect  // A2
      int_3 = 3
   EndSect  // ROOT 
"""
    return text


def test_search_keyword(pfs_ABC_text) -> None:
    pfs = mikeio.PfsDocument(StringIO(pfs_ABC_text))
    assert "A2" in pfs.ROOT

    r0 = pfs.search(key="not_there")
    assert len(r0) == 0

    r1 = pfs.search(key="float")
    assert r1.ROOT.A1.B.float_1 == 4.5
    assert "A2" not in r1.ROOT

    r2 = pfs.ROOT.search(key="float")
    assert r2.A1.B.float_1 == 4.5
    assert "A2" not in r2

    r3 = pfs.ROOT.search("float")
    assert r2 == r3


def test_search_keyword_found_in_multiple_places() -> None:
    pfs = mikeio.PfsDocument("tests/testdata/pfs/lake.sw")
    subset = pfs.search("charnock")
    # the string "Charnock" occurs 6 times in this file

    # NOTE: aliases not possible for search output
    SW = subset.FemEngineSW.SPECTRAL_WAVE_MODULE
    assert len(SW.WIND.keys()) == 2
    assert len(SW.OUTPUTS) == 4


def test_search_param(pfs_ABC_text) -> None:
    pfs = mikeio.PfsDocument(StringIO(pfs_ABC_text))

    r0 = pfs.search(param="not_there")
    assert len(r0) == 0

    r1 = pfs.search(param=0)
    assert len(r1.ROOT) == 2
    assert r1.ROOT.A1.int_1 == 0
    assert r1.ROOT.A2.int_2 == 0

    r2 = pfs.ROOT.search(param=0)
    assert r2 == r1.ROOT

    r3 = pfs.ROOT.search(param="0")
    assert len(r3) == 1
    assert r3.str_1 == "0"


def test_search_section(pfs_ABC_text) -> None:
    pfs = mikeio.PfsDocument(StringIO(pfs_ABC_text))

    r0 = pfs.search(section="not_there")
    assert len(r0) == 0

    r1 = pfs.search(section="A")
    assert len(r1.ROOT) == 2
    assert r1.ROOT.A1 == pfs.ROOT.A1
    assert r1.ROOT.A2 == pfs.ROOT.A2

    r2 = pfs.ROOT.search(section="A")
    assert r2 == r1.ROOT


def test_search_keyword_or_param(pfs_ABC_text) -> None:
    pfs = mikeio.PfsDocument(StringIO(pfs_ABC_text))
    # assert pfs.ROOT.A1.B.float_1 == 4.5

    r1 = pfs.search(key="float", param=3)
    assert r1.ROOT.A1.B.float_1 == 4.5
    assert r1.ROOT.int_3 == 3

    r2 = pfs.search(key="float")
    r3 = pfs.search(param=3)
    assert r1 != r2
    assert r1 != r3

    # r4 = mikeio.pfs._merge_PfsSections([r2, r3])
    # assert r1 == r4


def test_search_and_modify(pfs_ABC_text) -> None:
    # does the original remain un-changed?
    pfs = mikeio.PfsDocument(StringIO(pfs_ABC_text))
    assert pfs.ROOT.A1.B.float_1 == 4.5

    r1 = pfs.search(key="float")
    assert r1.ROOT.A1.B.float_1 == 4.5
    r1.ROOT.A1.B.float_1 = 99.9
    assert r1.ROOT.A1.B.float_1 == 99.9
    assert pfs.ROOT.A1.B.float_1 == 4.5


def test_clob_can_contain_pipe_characters() -> None:
    clob_text = """
    [WQRiverGroupPfs]
      Touched = false
      [WQRiverListPfs]
         Touched = false
         DisableProcesses = false
         [WQRiverPfs_0]
            Touched = false
            TemplateFile = ||
            Description = ''
            Clob = '<CLOB:22,1,1,false,1,0,"",0,"",0,"",0,"",0,"",0,"",0,"",0,"",||,false>'
         EndSect  // WQRiverPfs_0
      EndSect  // WQRiverListPfs
   EndSect  // WQRiverGroupPfs
   """
    pfs = mikeio.PfsDocument(StringIO(clob_text))
    sct = pfs.WQRiverGroupPfs.WQRiverListPfs.WQRiverPfs_0
    assert sct.Description == ""
    assert sct.TemplateFile == "||"
    assert (
        sct.Clob
        == '<CLOB:22,1,1,false,1,0,"",0,"",0,"",0,"",0,"",0,"",0,"",0,"",||,false>'
    )


def test_write_read_clob(tmp_path: Path) -> None:
    clob_text = """
    [WQRiverPfs_0]
        Clob = '<CLOB:22,1,1,false,1,0,"",0,"",0,"",0,"",0,"",0,"",0,"",0,"",||,false>'
    EndSect  // WQRiverPfs_0
   """
    pfs = mikeio.PfsDocument(StringIO(clob_text))
    sct = pfs.WQRiverPfs_0
    assert (
        sct.Clob
        == '<CLOB:22,1,1,false,1,0,"",0,"",0,"",0,"",0,"",0,"",0,"",0,"",||,false>'
    )

    fp = tmp_path / "clob.pfs"
    pfs.write(fp)

    pfs2 = mikeio.PfsDocument(fp)
    sct = pfs2.WQRiverPfs_0
    assert (
        sct.Clob
        == '<CLOB:22,1,1,false,1,0,"",0,"",0,"",0,"",0,"",0,"",0,"",0,"",||,false>'
    )


def test_ignores_comments_in_quotes() -> None:
    # Inspired by section from .mupp files

    text = """
      [SymbologyModule]
         SymbologyTreeView = '//'
      EndSect  // SymbologyModule
"""
    pfs = mikeio.PfsDocument.from_text(text)
    assert pfs.SymbologyModule.SymbologyTreeView == "//"


def test_filenames_may_contain_comma() -> None:
    pfs = mikeio.read_pfs("tests/testdata/pfs/tidal.21t")
    assert pfs.m21_tideph.Setup.File_1.mesh_file == "|.\\b,athy.mesh|"


def test_strip_comments() -> None:
    text = """
[Engine]
    var = 'x'  //  'y' (default)  🤖 ,.!//\\(foo,bar*) 😻
EndSect  // Engine
"""
    pfs = mikeio.PfsDocument.from_text(text)
    assert pfs.Engine.var == "x"


def test_get_nested_path() -> None:
    text = """
[foo]
    random = 1
    [bar]
        speed = 'fast'
        [baz]
            simple = 0
        EndSect
    EndSect
EndSect
"""

    pfs = mikeio.PfsDocument.from_text(text)
    assert pfs.foo.bar.baz.simple == 0
    assert pfs["foo"]["bar"]["baz"]["simple"] == 0
    path = "foo/bar/baz/simple"
    assert pfs[path] == 0

    part_of_the_path = "foo/bar"
    assert pfs[part_of_the_path].baz.simple == 0
    # we can mix and match
    assert pfs["foo"]["bar/baz/simple"] == 0


def test_set_nested_path() -> None:
    text = """
[foo]
    random = 1
    [bar]
        speed = 'fast'
        [baz]
            simple = 0
        EndSect
    EndSect
EndSect
"""

    pfs = mikeio.PfsDocument.from_text(text)
    assert pfs.foo.bar.baz.simple == 0
    path = "foo/bar/baz/simple"
    pfs[path] = 1
    assert pfs[path] == 1

    pfs.foo["bar/bat"] = {"n": 5, "s": "foo"}
    assert pfs.foo.bar.bat.n == 5
    assert pfs.foo.bar.bat.s == "foo"


def test_pfs_html_repr() -> None:
    """Test HTML representation for Jupyter notebooks."""
    text = """
[ENGINE]
    string_val = 'hello'
    int_val = 42
    float_val = 3.14
    bool_val = true
    list_val = 1, 2, 3
    [NESTED]
        nested_str = 'world'
        nested_num = 100
    EndSect  // NESTED
EndSect  // ENGINE
"""
    pfs = mikeio.PfsDocument.from_text(text)

    # Test that _repr_html_ method exists and returns HTML
    html = pfs.ENGINE._repr_html_()
    assert isinstance(html, str)
    assert len(html) > 0

    # Check for CSS styling
    assert "<style>" in html
    assert ".pfs-search-box" in html
    assert ".pfs-collapse-btn" in html

    # Check for proper HTML structure
    assert "<div id='pfs-" in html
    assert '<input type="text" class="pfs-search-box"' in html
    assert '<button class="pfs-collapse-btn">' in html

    # Check that values are present in the output
    assert "string_val" in html
    assert "int_val" in html
    assert "float_val" in html
    assert "bool_val" in html
    assert "NESTED" in html
    assert "nested_str" in html

    # Check for color-coded values
    assert "pfs-string" in html  # String values should be styled
    assert "pfs-number" in html  # Numeric values should be styled
    assert "pfs-bool" in html  # Boolean values should be styled
    assert "pfs-section-name" in html  # Section names should be styled

    # Check for collapsible sections
    assert 'type="checkbox"' in html
    assert "pfs-section-toggle" in html

    # Check for new features
    assert "pfs-collapse-btn" in html  # Collapse/expand all button
    assert "Collapse All" in html  # Button text
    assert "prefers-color-scheme: dark" in html  # Dark mode support
    assert "<script>" in html  # JavaScript for interactivity
    assert "setupCollapseExpand" in html  # Collapse/expand JavaScript function


def test_pfs_html_repr_nonunique_keys() -> None:
    """Test HTML repr with non-unique keys."""
    text = """
[ROOT]
    RGB_Color_Value = 128, 0, 128
    RGB_Color_Value = 85, 0, 171
EndSect  // ROOT
"""
    pfs = mikeio.PfsDocument(StringIO(text), unique_keywords=False)
    html = pfs.ROOT._repr_html_()

    assert isinstance(html, str)
    assert "RGB_Color_Value" in html
    assert "128" in html
    assert "171" in html


def test_pfs_html_repr_filepath() -> None:
    """Test HTML repr with file path highlighting."""
    text = """
[ENGINE]
    mesh_file = |.\\mesh.mesh|
    output_dir = '/tmp/output'
    number = 42
EndSect  // ENGINE
"""
    pfs = mikeio.PfsDocument.from_text(text)
    html = pfs.ENGINE._repr_html_()

    assert isinstance(html, str)
    # Check that file paths get special styling
    assert "pfs-filepath" in html
    # Check that both file path and regular string are present
    assert "mesh_file" in html
    assert "output_dir" in html
