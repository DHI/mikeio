import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import mikeio
    from mikeio import ItemInfo, EUMType, EUMUnit
    return EUMType, EUMUnit, ItemInfo, mikeio, np, pd


@app.cell
def _(mo):
    mo.md(
        r"""
        # Write a dfs0

        A mikeio.Dataset contains the information needed to write a dfs file. A Dataset consists of one or more mikeio.DataArrays each corresponding to an "item" in a dfs file.
        """
    )
    return


@app.cell
def _(
    EUMType,
    EUMUnit,
    ItemInfo,
    d1,
    da1,
    da2,
    item,
    mikeio,
    np,
    pd,
    time,
):
    nt = 10
    _time = pd.date_range('2000-1-1', periods=nt, freq='H')
    _d1 = np.zeros(nt)
    _item = ItemInfo('Zeros', EUMType.Water_Level)
    _da1 = mikeio.DataArray(d1, time=time, item=item)
    d2 = np.ones(nt)
    _item = ItemInfo('Ones', EUMType.Discharge, EUMUnit.meter_pow_3_per_sec)
    _da2 = mikeio.DataArray(d2, time=time, item=item)
    ds = mikeio.Dataset([da1, da2])
    ds
    return d2, ds, nt


@app.cell
def _(ds):
    ds.is_equidistant
    return


@app.cell
def _(ds):
    ds.to_dfs("test.dfs0", title="Zeros and ones")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Read a timeseries

        A dfs file is easily read with mikeio.read which returns a Dataset.
        """
    )
    return


@app.cell
def _(mikeio):
    ds_1 = mikeio.read('test.dfs0')
    ds_1
    return (ds_1,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## From comma separated file
        """
    )
    return


@app.cell
def _(pd):
    df = pd.read_csv("../tests/testdata/co2-mm-mlo.csv", parse_dates=True, index_col='Date', na_values=-99.99)
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Remove missing values
        """
    )
    return


@app.cell
def _(df):
    df_1 = df.dropna()
    df_1 = df_1[['Average', 'Trend']]
    df_1.plot()
    return (df_1,)


@app.cell
def _(mo):
    mo.md(
        r"""
        A dataframe with a datetimeindex can be used to create a dfs0 with a non-equidistant time axis by first converting it to a mikeio.Dataset.
        """
    )
    return


@app.cell
def _(df_1, mikeio):
    ds_2 = mikeio.from_pandas(df_1)
    ds_2
    return (ds_2,)


@app.cell
def _(mo):
    mo.md(
        r"""
        And then write to a dfs0 file:
        """
    )
    return


@app.cell
def _(ds_2):
    ds_2.to_dfs('mauna_loa_co2.dfs0')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        To get a equidistant time axis first interpolate to regularly spaced values, in this case daily.

        *The code for this can be written in many ways, below is an example, where we avoid temporary variables.*
        """
    )
    return


@app.cell
def _(df_1, mikeio):
    df_1.resample('D').interpolate().pipe(mikeio.from_pandas).to_dfs('mauna_loa_co2_daily.dfs0')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Read a timeseries
        """
    )
    return


@app.cell
def _(mikeio):
    res = mikeio.read("test.dfs0")
    res
    return (res,)


@app.cell
def _(res):
    res.time
    return


@app.cell
def _(res):
    res.to_numpy()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Or as a Pandas dataframe

        A mikeio.Dataset ds is converted to a pandas dataframe with ds.to_dataframe()
        """
    )
    return


@app.cell
def _(dfs0file, mikeio):
    _dfs0file = '../tests/testdata/da_diagnostic.dfs0'
    df_2 = mikeio.read(dfs0file).to_dataframe()
    df_2.head()
    return (df_2,)


@app.cell
def _(dfs0file, mikeio):
    _dfs0file = '../tests/testdata/random.dfs0'
    df_3 = mikeio.read(dfs0file).to_dataframe()
    df_3.head()
    return (df_3,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Create a timeseries with non-equidistant data
        """
    )
    return


@app.cell
def _(ItemInfo, d1, mikeio, np, pd, time):
    _d1 = np.random.uniform(low=0.0, high=5.0, size=5)
    _time = pd.DatetimeIndex(['2000-1-1', '2000-1-8', '2000-1-10', '2000-2-22', '2000-11-29'])
    da = mikeio.DataArray(d1, time=time, item=ItemInfo('Random'))
    da
    return (da,)


@app.cell
def _(da):
    da.is_equidistant
    return


@app.cell
def _(da):
    da.to_dfs("neq.dfs0", title="Non equidistant")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Create a timeseries with accumulated timestep
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Find correct eum units
        """
    )
    return


@app.cell
def _(EUMType):
    EUMType.search("prec")
    return


@app.cell
def _(EUMType):
    EUMType.Precipitation_Rate.units
    return


@app.cell
def _(EUMType, ItemInfo, da1, da2, item, mikeio, np, pd, time):
    from mikecore.DfsFile import DataValueType
    n = 1000
    _time = pd.date_range('2017-01-01 00:00', freq='H', periods=n)
    _item = ItemInfo(EUMType.Water_Level, data_value_type=DataValueType.Instantaneous)
    _da1 = mikeio.DataArray(data=np.random.random([n]), time=time, item=item)
    _item = ItemInfo('Nedbør', EUMType.Precipitation_Rate, data_value_type=DataValueType.Accumulated)
    _da2 = mikeio.DataArray(data=np.random.random([n]), time=time, item=item)
    ds_3 = mikeio.Dataset([da1, da2])
    ds_3.to_dfs('accumulated.dfs0')
    return DataValueType, ds_3, n


@app.cell
def _(mikeio):
    ds_4 = mikeio.read('accumulated.dfs0')
    ds_4
    return (ds_4,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Modify an existing timeseries
        """
    )
    return


@app.cell
def _(mikeio):
    ds_5 = mikeio.read('test.dfs0')
    ds_5
    return (ds_5,)


@app.cell
def _(ds_5):
    ds_5['Ones']
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Modify the data in some way...
        """
    )
    return


@app.cell
def _(ds_5, np):
    ds_5['Ones'] = ds_5['Ones'] * np.pi
    ds_5['Ones'].values
    return


@app.cell
def _(ds_5):
    ds_5.to_dfs('modified.dfs0')
    return


@app.cell
def _(mikeio):
    res_1 = mikeio.read('modified.dfs0')
    res_1['Ones']
    return (res_1,)


@app.cell
def _(mo):
    mo.md(
        r"""
        The second item is not modified.
        """
    )
    return


@app.cell
def _(res_1):
    res_1['Zeros']
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Convert units

        Read a file with waterlevel i meters.
        """
    )
    return


@app.cell
def _(filename, mikeio):
    _filename = '../tests/testdata/waterlevel_viken.dfs0'
    ds_6 = mikeio.read(filename)
    ds_6
    return (ds_6,)


@app.cell
def _(ds_6):
    ds_6.plot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The aim is to convert this timeseries to feet (1m = 3.3 ft)
        """
    )
    return


@app.cell
def _(ds_6):
    ds_6[0] = ds_6[0] * 3.3
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Which units are acceptable?
        """
    )
    return


@app.cell
def _(ds_6):
    ds_6.items[0].type.units
    return


@app.cell
def _(EUMUnit, ItemInfo, ds_6):
    ds_6[0].item = ItemInfo('Viken', ds_6[0].item.type, EUMUnit.feet)
    return


@app.cell
def _(ds_6):
    ds_6.to_dfs('wl_feet.dfs0')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ![WL](https://github.com/DHI/mikeio/raw/main/images/wl_feet.png)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Extrapolation
        """
    )
    return


@app.cell
def _(filename, mikeio):
    _filename = '../tests/testdata/waterlevel_viken.dfs0'
    ds_7 = mikeio.read(filename)
    df_4 = ds_7.to_dataframe()
    df_4.plot()
    return df_4, ds_7


@app.cell
def _(df_4, pd):
    rng = pd.date_range('1993-12-1', '1994-1-1', freq='30t')
    ix = pd.DatetimeIndex(rng)
    dfr = df_4.reindex(ix)
    dfr.plot()
    return dfr, ix, rng


@app.cell
def _(mo):
    mo.md(
        r"""
        Replace NaN with constant extrapolation (forward fill + back fill).
        """
    )
    return


@app.cell
def _(dfr):
    dfr_1 = dfr.ffill().bfill()
    dfr_1.plot()
    return (dfr_1,)


@app.cell
def _(dfr_1, ds_7):
    dfr_1.to_dfs0('Viken_extrapolated.dfs0', items=ds_7.items, title='Caution extrapolated data!')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Clean up
        """
    )
    return


@app.cell
def _():
    import os

    os.remove("test.dfs0")
    os.remove("modified.dfs0")
    os.remove("neq.dfs0")
    os.remove("accumulated.dfs0")
    os.remove("wl_feet.dfs0")
    os.remove("mauna_loa_co2_daily.dfs0")
    os.remove("mauna_loa_co2.dfs0")
    os.remove("Viken_extrapolated.dfs0")
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

