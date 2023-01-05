import os
from shutil import copyfile
import numpy as np
import pandas as pd
import mikeio
from mikeio import Dfsu
from mikeio import generic
from mikeio.generic import scale, diff, sum, extract, avg_time, fill_corrupt
import pytest


def test_add_constant(tmpdir):

    infilename = "tests/testdata/random.dfs0"
    outfilename = os.path.join(tmpdir.dirname, "add.dfs0")
    scale(infilename, outfilename, offset=100.0)

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue = org[0].values[0]
    expected = orgvalue + 100.0
    scaledvalue = scaled[0].values[0]
    assert scaledvalue == pytest.approx(expected)


def test_multiply_constant(tmpdir):

    infilename = "tests/testdata/random.dfs0"
    outfilename = os.path.join(tmpdir.dirname, "mult.dfs0")
    scale(infilename, outfilename, factor=1.5)

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue = org[0].values[0]
    expected = orgvalue * 1.5
    scaledvalue = scaled[0].values[0]
    assert scaledvalue == pytest.approx(expected)


def test_multiply_constant_single_item_number(tmpdir):

    infilename = "tests/testdata/wind_north_sea.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "mult.dfsu")
    scale(infilename, outfilename, factor=1.5, items=[0])

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue_speed = org[0].values[0][0]
    expected_speed = orgvalue_speed * 1.5
    scaledvalue_speed = scaled[0].values[0][0]
    assert scaledvalue_speed == pytest.approx(expected_speed)

    orgvalue_dir = org[1].values[0, 0]
    expected_dir = orgvalue_dir
    scaledvalue_dir = scaled[1].values[0, 0]
    assert scaledvalue_dir == pytest.approx(expected_dir)


def test_multiply_constant_single_item_name(tmpdir):

    infilename = "tests/testdata/wind_north_sea.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "multname.dfsu")
    scale(infilename, outfilename, factor=1.5, items=["Wind speed"])

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue_speed = org["Wind speed"].to_numpy()[0, 0]
    expected_speed = orgvalue_speed * 1.5
    scaledvalue_speed = scaled["Wind speed"].to_numpy()[0, 0]
    assert scaledvalue_speed == pytest.approx(expected_speed)

    orgvalue_dir = org["Wind direction"].to_numpy()[0, 0]
    expected_dir = orgvalue_dir
    scaledvalue_dir = scaled["Wind direction"].to_numpy()[0, 0]
    assert scaledvalue_dir == pytest.approx(expected_dir)


def test_diff_itself(tmpdir):

    infilename_1 = "tests/testdata/gebco_sound.dfs2"
    infilename_2 = "tests/testdata/gebco_sound.dfs2"
    outfilename = os.path.join(tmpdir.dirname, "diff.dfs2")

    diff(infilename_1, infilename_2, outfilename)

    org = mikeio.read(infilename_1)

    assert np.isnan(org["Elevation"].to_numpy()[0, -1, -1])

    diffed = mikeio.read(outfilename)

    diffedvalue = diffed["Elevation"].to_numpy()[0, 0, 0]
    assert diffedvalue == pytest.approx(0.0)
    assert np.isnan(diffed["Elevation"].to_numpy()[0, -1, -1])


def test_sum_itself(tmpdir):

    infilename_1 = "tests/testdata/gebco_sound.dfs2"
    infilename_2 = "tests/testdata/gebco_sound.dfs2"
    outfilename = os.path.join(tmpdir.dirname, "diff.dfs2")

    sum(infilename_1, infilename_2, outfilename)

    org = mikeio.read(infilename_1)

    summed = mikeio.read(outfilename)

    assert np.isnan(summed["Elevation"].to_numpy()[0][-1, -1])


def test_add_constant_delete_values_unchanged(tmpdir):

    infilename = "tests/testdata/gebco_sound.dfs2"
    outfilename = os.path.join(tmpdir.dirname, "adj.dfs2")
    scale(infilename, outfilename, offset=-2.1, items=["Elevation"])

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue = org["Elevation"].to_numpy()[0, 0, 0]
    scaledvalue = scaled["Elevation"].to_numpy()[0, 0, 0]
    assert scaledvalue == pytest.approx(orgvalue - 2.1)

    orgvalue = org["Elevation"].to_numpy()[0, 100, 0]
    assert np.isnan(orgvalue)

    scaledvalue = scaled["Elevation"].to_numpy()[0, 100, 0]
    assert np.isnan(scaledvalue)


def test_multiply_constant_delete_values_unchanged_2(tmpdir):

    infilename = "tests/testdata/random_two_item.dfs2"
    outfilename = os.path.join(tmpdir.dirname, "adj.dfs2")

    item_name = "testing water level"

    scale(infilename, outfilename, factor=1000.0, items=[item_name])

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue = org[item_name].to_numpy()[0, 0, 0]
    scaledvalue = scaled[item_name].to_numpy()[0, 0, 0]
    assert scaledvalue == pytest.approx(orgvalue * 1000.0)

    orgvalue = org[item_name].to_numpy()[0, -11, 0]
    assert np.isnan(orgvalue)

    scaledvalue = scaled[item_name].to_numpy()[0, -11, 0]
    assert np.isnan(scaledvalue)


def test_linear_transform(tmpdir):

    infilename = "tests/testdata/random.dfs0"
    outfilename = os.path.join(tmpdir.dirname, "linear.dfs0")
    scale(infilename, outfilename, offset=-20.0, factor=1.5)

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue = org[0].values[0]
    expected = orgvalue * 1.5 - 20.0
    scaledvalue = scaled[0].values[0]
    assert scaledvalue == pytest.approx(expected)


def test_linear_transform_dfsu(tmpdir):

    infilename = "tests/testdata/HD2D.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "linear.dfsu")
    scale(infilename, outfilename, offset=-20.0, factor=1.5)

    org = mikeio.read(infilename)

    scaled = mikeio.read(outfilename)

    orgvalue = org[0].values[0]
    expected = orgvalue * 1.5 - 20.0
    scaledvalue = scaled[0].values[0]
    assert scaledvalue == pytest.approx(expected)


def test_sum_dfsu(tmpdir):

    infilename_a = "tests/testdata/HD2D.dfsu"
    infilename_b = "tests/testdata/HD2D.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "sum.dfsu")
    mikeio.generic.sum(infilename_a, infilename_b, outfilename)

    org = mikeio.read(infilename_a)

    summed = mikeio.read(outfilename)

    orgvalue = org[0].values[0]
    expected = orgvalue * 2
    scaledvalue = summed[0].values[0]
    assert scaledvalue == pytest.approx(expected)


def test_diff_dfsu(tmpdir):

    infilename_a = "tests/testdata/HD2D.dfsu"
    infilename_b = "tests/testdata/HD2D.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "diff.dfsu")
    mikeio.generic.diff(infilename_a, infilename_b, outfilename)

    org = mikeio.read(infilename_a)

    diffed = mikeio.read(outfilename)

    expected = 0.0
    scaledvalue = diffed[0].values[0]
    assert scaledvalue == pytest.approx(expected)


def test_concat_overlapping(tmpdir):
    infilename_a = "tests/testdata/tide1.dfs1"
    infilename_b = "tests/testdata/tide2.dfs1"
    outfilename = os.path.join(tmpdir.dirname, "concat.dfs1")

    mikeio.generic.concat([infilename_a, infilename_b], outfilename)

    ds = mikeio.read(outfilename)
    assert len(ds.time) == 145


def test_concat_files_gap_fail(tmpdir):
    infilename_a = "tests/testdata/tide1.dfs1"
    infilename_b = "tests/testdata/tide4.dfs1"
    outfilename = os.path.join(tmpdir.dirname, "concat.dfs1")
    with pytest.raises(Exception):
        mikeio.generic.concat([infilename_a, infilename_b], outfilename)


def test_concat_three_files(tmpdir):
    infiles = [
        "tests/testdata/tide1.dfs1",
        "tests/testdata/tide2.dfs1",
        "tests/testdata/tide4.dfs1",
    ]
    outfilename = os.path.join(tmpdir.dirname, "concat.dfs1")

    mikeio.generic.concat(infiles, outfilename)

    ds = mikeio.read(outfilename)
    assert len(ds.time) == (5 * 48 + 1)


def test_concat_closes_files(tmpdir):
    infiles = [
        "tests/testdata/tide1.dfs1",
        "tests/testdata/tide2.dfs1",
        "tests/testdata/tide4.dfs1",
    ]

    tmpfiles = []

    for file in infiles:
        tmp_path = os.path.join(tmpdir.dirname, os.path.basename(file))
        copyfile(file, tmp_path)
        tmpfiles.append(tmp_path)

    outfilename = os.path.join(tmpdir.dirname, "concat.dfs1")

    mikeio.generic.concat(tmpfiles, outfilename)

    for file in tmpfiles:
        os.remove(file)
        assert not os.path.exists(file)

    # ds = mikeio.read(outfilename)
    # assert len(ds.time) == (5 * 48 + 1)


def test_concat_keep(tmpdir):
    """
    test keep arguments of concatenation function
    """
    # added keep arguments to test
    keep_args = ["first", "last"]

    infiles = [
        "tests/testdata/tide1.dfs1",
        "tests/testdata/tide2_offset.dfs1",
        "tests/testdata/tide4.dfs1",
    ]
    outfilename = os.path.join(tmpdir.dirname, "concat.dfs1")

    # loop through keep args to test all
    for keep_arg in keep_args:
        mikeio.generic.concat(
            infilenames=infiles, outfilename=outfilename, keep=keep_arg
        )

        # read outfile
        dso = mikeio.read(outfilename)
        df_o = pd.DataFrame(dso[0].to_numpy(), index=dso.time)

        """ handle data (always keep two data in memory to compare in overlapping period) """
        # !!! checks below solely consider one item !!
        for i, infile in enumerate(infiles):
            dsi = mikeio.read(infile)

            # just store current (=last) data in first loop. no check required
            if i == 0:
                df_last = pd.DataFrame(dsi[0].to_numpy(), index=dsi.time)
            else:
                # move previously last to first
                df_first = df_last.copy()
                # overwrite last with new / current data
                df_last = pd.DataFrame(dsi[0].to_numpy(), index=dsi.time)

                # find overlapping timesteps via merge
                overlap_idx = pd.merge(
                    df_first, df_last, how="inner", left_index=True, right_index=True
                ).index

                # check if first and output are identical (results in dataframe with overlapping datetime index and bool values)
                first_out = (
                    (df_first.loc[overlap_idx] == df_o.loc[overlap_idx])
                    .eq(True)
                    .all()
                    .all()
                )
                last_out = (
                    (df_last.loc[overlap_idx] == df_o.loc[overlap_idx])
                    .eq(True)
                    .all()
                    .all()
                )

                if keep_arg == "first":
                    assert first_out, "overlap should be with first dataset"
                elif keep_arg == "last":
                    assert last_out, "overlap should be with last dataset"


def test_extract_equidistant(tmpdir):

    infile = "tests/testdata/waves.dfs2"
    outfile = os.path.join(tmpdir.dirname, "waves_subset.dfs2")

    extract(infile, outfile, start=1, end=-1)
    orig = mikeio.read(infile)
    extracted = mikeio.read(outfile)
    assert extracted.n_timesteps == orig.n_timesteps - 1
    assert extracted.time[0] == orig.time[1]

    with pytest.raises(ValueError):
        extract(infile, outfile, start=-23, end=1000)

    with pytest.raises(ValueError):
        extract(infile, outfile, start=-23.5)

    with pytest.raises(ValueError):
        extract(infile, outfile, start=1000)


def test_extract_non_equidistant(tmpdir):

    infile = "tests/testdata/da_diagnostic.dfs0"
    outfile = os.path.join(tmpdir.dirname, "diagnostic_subset.dfs0")

    extract(infile, outfile, start="2017-10-27 01:58", end="2017-10-27 04:32")
    extracted = mikeio.read(outfile)
    assert extracted.n_timesteps == 16
    assert extracted.time[0].hour == 2

    extract(infile, outfile, end=3600.0)
    extracted = mikeio.read(outfile)
    assert extracted.n_timesteps == 7
    assert extracted.time[0].hour == 0
    assert extracted.time[-1].minute == 0

    with pytest.raises(ValueError):
        extract(infile, outfile, start=1800.0, end=237981231.232)

    extract(infile, outfile, "2017-10-27,2017-10-28")
    extracted = mikeio.read(outfile)
    assert extracted.n_timesteps > 0

    with pytest.raises(ValueError):
        extract(infile, outfile, start=7200.0, end=1800.0)


def test_extract_relative_time_axis(tmpdir):

    infile = "tests/testdata/eq_relative.dfs0"
    outfile = os.path.join(tmpdir.dirname, "eq_relative_extract.dfs0")

    with pytest.raises(Exception):
        extract(infile, outfile, start=0, end=4)


def test_extract_step_equidistant(tmpdir):

    infile = "tests/testdata/tide1.dfs1"  # 30min
    outfile = os.path.join(tmpdir.dirname, "tide1_6hour.dfs1")

    extract(infile, outfile, step=12)
    orig = mikeio.read(infile)
    extracted = mikeio.read(outfile)
    assert extracted.n_timesteps == 9
    assert extracted.timestep == 6 * 3600
    assert extracted.time[0] == orig.time[0]
    assert extracted.time[-1] == orig.time[-1]


def test_extract_step_non_equidistant(tmpdir):

    infile = "tests/testdata/da_diagnostic.dfs0"
    outfile = os.path.join(tmpdir.dirname, "diagnostic_step3.dfs0")

    extract(infile, outfile, step=3)
    orig = mikeio.read(infile)
    extracted = mikeio.read(outfile)
    assert extracted.n_timesteps * 3 == orig.n_timesteps
    assert extracted.time[0] == orig.time[0]
    assert extracted.time[-1] == orig.time[-3]


def test_extract_items(tmpdir):

    # This is a Dfsu 3d file (i.e. first item is Z coordinate)
    infile = "tests/testdata/oresund_vertical_slice.dfsu"
    outfile = os.path.join(tmpdir.dirname, "extracted_vertical_slice.dfsu")

    extract(infile, outfile, items="Temperature")
    extracted = mikeio.read(outfile)
    assert extracted.n_items == 1
    assert extracted.items[0].name == "Temperature"

    extract(infile, outfile, items=[1])
    extracted = mikeio.read(outfile)
    assert extracted.n_items == 1
    assert extracted.items[0].name == "Salinity"

    extract(infile, outfile, items=range(0, 2))  # [0,1]
    extracted = mikeio.read(outfile)
    assert extracted.n_items == 2
    assert extracted.items[0].name == "Temperature"

    extract(infile, outfile, items=["Salinity", 0])
    extracted = mikeio.read(outfile)
    assert extracted.n_items == 2

    with pytest.raises(Exception):
        # must be unique
        extract(infile, outfile, items=["Salinity", 1])

    with pytest.raises(Exception):
        # no negative numbers
        extract(infile, outfile, items=[0, 2, -1])

    with pytest.raises(Exception):
        extract(infile, outfile, items=[0, "not_an_item"])


def test_time_average(tmpdir):

    infilename = "tests/testdata/NorthSea_HD_and_windspeed.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "NorthSea_HD_and_windspeed_avg.dfsu")
    avg_time(infilename, outfilename)

    org = mikeio.read(infilename)

    averaged = mikeio.read(outfilename)

    assert all([a == b for a, b in zip(org.items, averaged.items)])
    assert org.time[0] == averaged.time[0]
    assert org.shape[1] == averaged.shape[1]
    assert averaged.shape[0] == 1
    assert np.allclose(org.mean(axis=0)[0].to_numpy(), averaged[0].to_numpy())


def test_time_average_dfsu_3d(tmpdir):
    infilename = "tests/testdata/oresund_sigma_z.dfsu"
    outfilename = os.path.join(tmpdir, "oresund_sigma_z_avg.dfsu")
    avg_time(infilename, outfilename)

    org = mikeio.open(infilename)
    averaged = mikeio.open(outfilename)

    assert averaged.n_timesteps == 1
    assert org.start_time == averaged.start_time
    assert org.n_items == averaged.n_items


def test_time_average_deletevalues(tmpdir):

    infilename = "tests/testdata/gebco_sound.dfs2"
    outfilename = os.path.join(tmpdir.dirname, "gebco_sound_avg.dfs2")
    avg_time(infilename, outfilename)

    org = mikeio.read(infilename)
    averaged = mikeio.read(outfilename)

    assert all([a == b for a, b in zip(org.items, averaged.items)])
    assert org.time[0] == averaged.time[0]
    assert org.shape[1] == averaged.shape[1]
    nan1 = np.isnan(org[0].to_numpy())
    nan2 = np.isnan(averaged[0].to_numpy())
    assert np.all(nan1 == nan2)
    assert np.allclose(org[0].to_numpy()[~nan1], averaged[0].to_numpy()[~nan2])


def test_quantile_dfsu(tmpdir):

    infilename = "tests/testdata/oresundHD_run1.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "oresund_q10.dfsu")
    generic.quantile(infilename, outfilename, q=0.1, items=["Surface elevation"])

    org = mikeio.read(infilename).quantile(q=0.1, axis=0)
    q10 = mikeio.read(outfilename)

    assert np.allclose(org[0].to_numpy(), q10[0].to_numpy())


def test_quantile_dfsu_buffer_size(tmpdir):

    infilename = "tests/testdata/oresundHD_run1.dfsu"
    outfilename = os.path.join(tmpdir.dirname, "oresund_q10.dfsu")
    generic.quantile(infilename, outfilename, q=0.1, buffer_size=1e5, items=0)

    org = mikeio.read(infilename).quantile(q=0.1, axis=0)
    q10 = mikeio.read(outfilename)

    assert np.allclose(org[0].to_numpy(), q10[0].to_numpy())


def test_quantile_dfs2(tmpdir):

    infilename = "tests/testdata/eq.dfs2"
    outfilename = os.path.join(tmpdir.dirname, "eq_q90.dfs2")
    generic.quantile(infilename, outfilename, q=0.9)

    org = mikeio.read(infilename).quantile(q=0.9, axis=0)
    q90 = mikeio.read(outfilename)

    assert np.allclose(org[0].to_numpy(), q90[0].to_numpy())


def test_quantile_dfs0(tmpdir):

    infilename = "tests/testdata/da_diagnostic.dfs0"
    outfilename = os.path.join(tmpdir.dirname, "da_q001_q05.dfs0")
    generic.quantile(infilename, outfilename, q=[0.01, 0.5])

    org = mikeio.read(infilename).quantile(q=[0.01, 0.5], axis=0)
    qnt = mikeio.read(outfilename)

    assert np.allclose(org[0].to_numpy(), qnt[0].to_numpy())
    # assert np.allclose(org[5], qnt[5])


def test_quantile_dfsu_3d(tmpdir):
    infilename = "tests/testdata/oresund_sigma_z.dfsu"
    outfilename = os.path.join(tmpdir, "oresund_sigma_z_avg.dfsu")
    generic.quantile(infilename, outfilename, q=[0.1,0.9], items=["Temperature"])

    qd = mikeio.open(outfilename)
    assert qd.n_timesteps == 1


def test_dfs_ext_capitalisation(tmpdir):
    filename = os.path.join("tests", "testdata", "waves2.DFS0")
    ds = mikeio.open(filename)
    ds = mikeio.read(filename)
    ds.to_dfs(os.path.join(tmpdir, "void.DFS0"))
    filename = os.path.join("tests", "testdata", "odense_rough2.MESH")
    ds = mikeio.open(filename)
    filename = os.path.join("tests", "testdata", "oresund_vertical_slice2.DFSU")
    ds = mikeio.open(filename)
    assert True


def test_fill_corrupt_data(tmpdir):
    """This test doesn't verify much..."""

    infile = "tests/testdata/waves.dfs2"

    outfile = os.path.join(tmpdir.dirname, "waves_subset.dfs2")

    fill_corrupt(infilename=infile, outfilename=outfile, items=[1])
    orig = mikeio.read(infile)
    extracted = mikeio.read(outfile)
    assert extracted.n_timesteps == orig.n_timesteps
