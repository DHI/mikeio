import os
from datetime import datetime

from mikeio import Pfs
from mikeio.pfs import PfsCore


def test_basic():

    pfs = Pfs("tests/testdata/simple.pfs")

    data = pfs.data
    # On a pfs file with a single target, the target is implicit,
    #  i.e. BoundaryExtractor in this case

    assert data.z_min == -3000
    assert data.POINT_1.y == 50


def test_sw():

    pfs = Pfs("tests/testdata/lake.sw")
    data = pfs.data

    # On a pfs file with a single target, the target is implicit,
    #  i.e. FemEngineSW in this case

    assert data.SPECTRAL_WAVE_MODULE.SPECTRAL.number_of_frequencies == 25
    assert data.SPECTRAL_WAVE_MODULE.SPECTRAL.number_of_frequencies == 25

    # use shorthand alias SW instead of SPECTRAL_WAVE_MODULE
    assert data.SW.SPECTRAL.number_of_frequencies == 25
    assert data.SW.WIND.format == 1

    assert data.TIME.number_of_time_steps == 450

    assert data.TIME.start_time.year == 2002
    assert data.TIME.start_time.month == 1


def test_outputs():

    pfs = Pfs("tests/testdata/lake.sw")
    df = pfs.get_outputs(section="SPECTRAL_WAVE_MODULE")

    assert df["file_name"][1] == "Wave_parameters.dfsu"


def test_sw_outputs():

    pfs = Pfs("tests/testdata/lake.sw")
    df = pfs.data.SW.get_outputs()

    assert df["file_name"][1] == "Wave_parameters.dfsu"
    assert df.shape[0] == 4

    df = pfs.data.SW.get_outputs(included_only=True)

    assert df["file_name"][1] == "Wave_parameters.dfsu"
    assert df.shape[0] == 3


def test_hd_outputs():

    pfs = Pfs("tests/testdata/lake.m21fm")
    df = pfs.data.HD.get_outputs()

    assert df["file_name"][2] == "ts.dfs0"
    assert df.shape[0] == 3

    df = pfs.data.HD.get_outputs(included_only=True)

    assert df.shape[0] == 2


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


## PFSCore wrapping DHI.PFS.PFSFile


def test_sw_new_start_time_write(tmpdir):

    pfs = PfsCore("tests/testdata/lake.sw", "FemEngineSW")

    new_start = datetime(2020, 6, 10, 12)

    assert pfs.start_time.year == 2002
    pfs.start_time = new_start

    assert pfs.start_time.year == 2020

    outfilename = os.path.join(tmpdir.dirname, "lake_mod.sw")

    pfs.write(outfilename)


def test_sw_get_end_time(tmpdir):

    pfs = PfsCore("tests/testdata/lake.sw", "FemEngineSW")

    assert pfs.start_time.year == 2002
    assert pfs.end_time.year == 2002
    assert pfs.end_time.month == 1
    assert pfs.end_time.day == 1
    assert pfs.end_time.hour == 15


def test_sw_set_end_time(tmpdir):

    pfs = PfsCore("tests/testdata/lake.sw", "FemEngineSW")

    assert pfs.start_time.year == 2002
    assert pfs.section("TIME")["number_of_time_steps"].value == 450
    new_end_time = datetime(2002, 1, 2, 0)
    pfs.end_time = new_end_time

    assert pfs.end_time == new_end_time
    assert pfs.section("TIME")["number_of_time_steps"].value == 720


def test_sw_modify_and_write(tmpdir):

    # [SPECTRAL_WAVE_MODULE][WIND]Charnock_parameter
    pfs = PfsCore("tests/testdata/lake.sw")

    wind_section = pfs.section("SPECTRAL_WAVE_MODULE").section("WIND")

    assert type(wind_section["Charnock_parameter"].value) == float

    # modify parameter without needing to specify that the float is a float...
    wind_section["Charnock_parameter"] = 0.02

    # modify a filename
    pfs.section("DOMAIN")["file_name"] = "tests\\testdata\\odense_rough.mesh"

    # attribute access
    out_file_1 = (
        pfs.section("SPECTRAL_WAVE_MODULE")
        .section("OUTPUTS")
        .section("OUTPUT_1")
        .file_name
    )

    assert out_file_1.value == "Wave_parameters.dfsu"

    outfilename = os.path.join(tmpdir.dirname, "lake_mod.sw")

    pfs.write(outfilename)

