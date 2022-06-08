# Dfs0

A dfs0 file is also called a time series file.

Working with data from dfs0 files are conveniently done in one of two ways:

* [`mikeio.Dataset`](Dataset) - keeps EUM information (convenient if you save data to new dfs0 file)
* [`pandas.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) - utilize all the powerful methods of pandas


## Read Dfs0 to Dataset


```python
>>> import mikeio
>>> ds = mikeio.read("da_diagnostic.dfs0")
>>> ds
<mikeio.Dataset>
dims: (time:744)
time: 2017-10-27 00:00:00 - 2017-10-29 18:00:00 (744 non-equidistant records)
items:
  0:  State 1Sign. Wave Height <Significant wave height> (meter)
  1:  State 2Sign. Wave Height <Significant wave height> (meter)
  2:  Mean StateSign. Wave Height <Significant wave height> (meter)
  3:  MeasurementSign. Wave Height <Significant wave height> (meter)
```

## From Dfs0 to pandas DataFrame

```python
>>> df = ds.to_dataframe()
>>> df.head()
                     State 1Sign. Wave Height  State 2Sign. Wave Height  Mean StateSign. Wave Height  MeasurementSign. Wave Height
2017-10-27 00:00:00                  1.749465                  1.749465                     1.749465                          1.72
2017-10-27 00:10:00                  1.811340                  1.796895                     1.807738                           NaN
2017-10-27 00:20:00                  1.863424                  1.842759                     1.853422                           NaN
2017-10-27 00:30:00                  1.922261                  1.889839                     1.897670                           NaN
2017-10-27 00:40:00                  1.972455                  1.934886                     1.935281                           NaN

```

## From pandas DataFrame to Dfs0

```python
>>> import mikeio
>>> df = pd.read_csv("co2-mm-mlo.csv", parse_dates=True, index_col='Date', na_values=-99.99)
>>> df.to_dfs0("mauna_loa_co2.dfs0")
```

## Dfs0 example notebooks

* [Dfs0](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs0%20-%20Timeseries.ipynb) - read, write, to_dataframe, non-equidistant, accumulated timestep, extrapolation
* [Dfs0 Relative-time](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs0%20-%20Relative%20time.ipynb) - read file with relative time axis
* [Dfs0 | getting-started-with-mikeio](https://dhi.github.io/getting-started-with-mikeio/dfs0.html) - Course literature




## Dfs0 API

```{eval-rst}
.. autoclass:: mikeio.Dfs0
	:members:
	:inherited-members:
```