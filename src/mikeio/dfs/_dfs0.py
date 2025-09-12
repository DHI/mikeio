from __future__ import annotations
from functools import cached_property
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd
from mikecore.DfsFactory import DfsBuilder, DfsFactory
from mikecore.DfsFile import DfsSimpleType, StatType, TimeAxisType
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.eum import eumQuantity

from .. import __dfs_version__
from ..dataset import Dataset, DataArray
from ._dfs import _get_item_info, _valid_item_numbers
from ..eum import EUMType, EUMUnit, ItemInfo, TimeStepUnit, ItemInfoList
from .._time import DateTimeSelector


def write_dfs0(
    filename: str | Path,
    dataset: Dataset,
    title: str = "",
    dtype: DfsSimpleType | np.float32 | np.float64 = DfsSimpleType.Float,
) -> None:
    filename = str(filename)

    factory = DfsFactory()
    builder = DfsBuilder.Create(title, "mikeio", __dfs_version__)
    builder.SetDataType(1)
    builder.SetGeographicalProjection(factory.CreateProjectionUndefined())

    system_start_time = dataset.time[0]

    if dataset.is_equidistant:
        if len(dataset.time) == 1:
            dt = 1.0  # TODO
        else:
            dt = (dataset.time[1] - dataset.time[0]).total_seconds()

        temporal_axis = factory.CreateTemporalEqCalendarAxis(
            TimeStepUnit.SECOND, system_start_time, 0, dt
        )
    else:
        temporal_axis = factory.CreateTemporalNonEqCalendarAxis(
            TimeStepUnit.SECOND, system_start_time
        )

    builder.SetTemporalAxis(temporal_axis)
    builder.SetItemStatisticsType(StatType.RegularStat)

    dfs_dtype = Dfs0._to_dfs_datatype(dtype)

    for da in dataset:
        newitem = builder.CreateDynamicItemBuilder()
        quantity = eumQuantity.Create(da.type, da.unit)
        newitem.Set(da.name, quantity, dfs_dtype)
        newitem.SetValueType(da.item.data_value_type)
        newitem.SetAxis(factory.CreateAxisEqD0())
        builder.AddDynamicItem(newitem.GetDynamicItemInfo())

    builder.CreateFile(filename)

    dfs = builder.GetFile()

    delete_value = dfs.FileInfo.DeleteValueFloat

    t_seconds = (dataset.time - dataset.time[0]).total_seconds().values

    data = np.array(dataset.to_numpy(), order="F").astype(np.float64).T
    data[np.isnan(data)] = delete_value

    if data.ndim == 2:
        data_to_write = np.concatenate([t_seconds.reshape(-1, 1), data], axis=1)
    else:
        data_to_write = np.concatenate(
            [np.atleast_2d(t_seconds), np.atleast_2d(data)], axis=1
        )
    dfs.WriteDfs0DataDouble(data_to_write)

    dfs.Close()


class Dfs0:
    """Class for reading/writing dfs0 files."""

    def __init__(self, filename: str | Path):
        """Create a Dfs0 object for reading, writing.

        Parameters
        ----------
        filename: str or Path
            File name including full path to the dfs0 file.

        """
        self._filename = str(filename)

        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(path)

        dfs = DfsFileFactory.DfsGenericOpen(str(path))
        self._dfs = dfs

        # Read items
        self._items = _get_item_info(dfs.ItemInfo)

        self._timeaxistype = dfs.FileInfo.TimeAxis.TimeAxisType

        if self._timeaxistype in {
            TimeAxisType.CalendarEquidistant,
            TimeAxisType.CalendarNonEquidistant,
        }:
            self._start_time: datetime = dfs.FileInfo.TimeAxis.StartDateTime
        else:  # relative time axis
            self._start_time = datetime(1970, 1, 1)

        # time
        self._n_timesteps: int = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

        dfs.Close()

    def __repr__(self) -> str:
        out = ["<mikeio.Dfs0>"]
        out.append(f"timeaxis: {repr(self._timeaxistype)}")

        if self.n_items < 10:
            out.append("items:")
            for i, item in enumerate(self.items):
                out.append(f"  {i}:  {item}")
        else:
            out.append(f"number of items: {self.n_items}")

        return str.join("\n", out)

    def read(
        self,
        items: str | int | Sequence[str | int] | None = None,
        time: int | str | slice | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """Read data from a dfs0 file.

        Parameters
        ----------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time: int, str, datetime, pd.TimeStamp, sequence, slice or pd.DatetimeIndex, optional
            Read only selected time steps, by default None (=all)
        **kwargs: Any
            Additional keyword arguments are ignored

        Returns
        -------
        Dataset
            A Dataset with data dimensions [t]

        """
        path = Path(self._filename)
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found")

        # read data from file
        dfs = DfsFileFactory.DfsGenericOpen(self._filename)
        raw_data = dfs.ReadDfs0DataDouble()  # Bulk read the data
        dfs.Close()

        matrix = raw_data[:, 1:]
        matrix[matrix == dfs.FileInfo.DeleteValueDouble] = np.nan  # cutil
        matrix[matrix == dfs.FileInfo.DeleteValueFloat] = np.nan  # linux
        ncol = matrix.shape[1]
        data = [matrix[:, i] for i in range(ncol)]

        t_seconds = raw_data[:, 0]
        ftime = pd.to_datetime(t_seconds, unit="s", origin=self.start_time).round(
            freq="ms"
        )

        # TODO common for all dfs files , extract
        # select items
        item_numbers = _valid_item_numbers(dfs.ItemInfo, items)
        if items is not None:
            data = [data[it] for it in item_numbers]
            item_infos = [self.items[it] for it in item_numbers]
        else:
            item_infos = self.items
        ds = Dataset.from_numpy(data, time=ftime, items=item_infos, validate=False)

        # select time steps
        if self._timeaxistype == TimeAxisType.CalendarNonEquidistant and isinstance(
            time, str
        ):
            return ds.sel(time=time)

        else:
            time_steps = None
            if time is not None:
                if isinstance(time, slice) and isinstance(time.start, str):
                    return ds.sel(time=time)
                else:
                    dts = DateTimeSelector(ftime)
                    time_steps = dts.isel(time)

        if time_steps:
            ds = ds.isel(time=time_steps)

        return ds

    @staticmethod
    def _to_dfs_datatype(
        dtype: DfsSimpleType | np.float32 | np.float64,
    ) -> DfsSimpleType:
        if dtype in {np.float64, DfsSimpleType.Double}:
            return DfsSimpleType.Double

        if dtype in {np.float32, DfsSimpleType.Float}:
            return DfsSimpleType.Float

        raise TypeError("Dfs files only support float or double")

    def to_dataframe(self, unit_in_name: bool = False) -> pd.DataFrame:
        """Read data from the dfs0 file and return a Pandas DataFrame.

        Parameters
        ----------
        unit_in_name: bool, optional
            include unit in column name, default False
        Returns
        -------
        pd.DataFrame

        """
        ds = self.read()
        df = ds.to_dataframe()

        if unit_in_name:
            mapping = {
                item.name: f"{item.name} ({item.unit.name})" for item in ds.items
            }
            df = df.rename(columns=mapping)

        return df

    @property
    def n_items(self) -> int:
        """Number of items."""
        return len(self._items)

    @property
    def items(self) -> ItemInfoList:
        """List of items."""
        return self._items

    @property
    def start_time(self) -> datetime:
        """File start time."""
        return self._start_time

    @cached_property
    def end_time(self) -> datetime:
        if self._dfs.FileInfo.TimeAxis.IsEquidistant():
            dt = self._dfs.FileInfo.TimeAxis.TimeStep
            n_steps = self._dfs.FileInfo.TimeAxis.NumberOfTimeSteps
            timespan = dt * (n_steps - 1)
        else:
            timespan = self._dfs.FileInfo.TimeAxis.TimeSpan

        return self.start_time + timedelta(seconds=timespan)

    @property
    def n_timesteps(self) -> int:
        """Number of time steps."""
        return self._n_timesteps

    @property
    def timestep(self) -> float:
        """Time step size in seconds."""
        if self._timeaxistype == TimeAxisType.CalendarEquidistant:
            return self._dfs.FileInfo.TimeAxis.TimeStep  # type: ignore
        else:
            raise ValueError("Time axis type not supported")

    @property
    def time(self) -> pd.DatetimeIndex:
        """File all datetimes."""
        if self._timeaxistype == TimeAxisType.CalendarEquidistant:
            freq = pd.Timedelta(seconds=self.timestep)
            return pd.date_range(
                start=self.start_time,
                periods=self.n_timesteps,
                freq=freq,
            )

        elif self._timeaxistype == TimeAxisType.CalendarNonEquidistant:
            return self.read().time
        else:
            raise ValueError("Time axis type not supported")

    # ======================
    # Deprecated in 2.5.0
    # ======================

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        filename: str | Path,
        itemtype: EUMType | None = None,
        unit: EUMUnit | None = None,
        items: Sequence[ItemInfo] | None = None,
    ) -> None:
        return dataframe_to_dfs0(df, filename, itemtype, unit, items)


def series_to_dfs0(
    self: pd.Series,
    filename: str,
    itemtype: EUMType | None = None,
    unit: EUMUnit | None = None,
    items: Sequence[ItemInfo] | None = None,
    title: str | None = None,
    dtype: Any = DfsSimpleType.Float,
) -> None:
    df = pd.DataFrame(self)
    df.to_dfs0(filename, itemtype, unit, items, title, dtype)


def dataframe_to_dfs0(
    self: pd.DataFrame,
    filename: str | Path,
    itemtype: EUMType | None = None,
    unit: EUMUnit | None = None,
    items: Sequence[ItemInfo] | None = None,
    title: str = "",
    dtype: Any = DfsSimpleType.Float,
) -> None:
    warnings.warn(
        "series/dataframe_to_dfs0 is deprecated. Use mikeio.from_pandas instead.",
        FutureWarning,
    )
    if not isinstance(self.index, pd.DatetimeIndex):
        raise ValueError(
            "Dataframe index must be a DatetimeIndex. Hint: pd.read_csv(..., parse_dates=True)"
        )

    ncol = self.values.shape[1]
    data = [self.values[:, i] for i in range(ncol)]

    if items is None:
        if itemtype is None:
            items = [ItemInfo(name) for name in self.columns]
        else:
            if unit is None:
                items = [ItemInfo(name, itemtype) for name in self.columns]
            else:
                items = [ItemInfo(name, itemtype, unit) for name in self.columns]

    das = {
        item.name: DataArray(data=d, item=item, time=self.index)
        for d, item in zip(data, items)
    }
    ds = Dataset(das)
    write_dfs0(filename=filename, dataset=ds, title=title, dtype=dtype)


# Monkey patching onto Pandas classes
pd.DataFrame.to_dfs0 = dataframe_to_dfs0

pd.Series.to_dfs0 = series_to_dfs0
