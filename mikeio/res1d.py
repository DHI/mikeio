import os.path

import clr
import pandas as pd

from mikeio.dotnet import from_dotnet_datetime

clr.AddReference("DHI.Mike1D.ResultDataAccess")
from DHI.Mike1D.ResultDataAccess import ResultData, ResultDataQuery  # noqa

clr.AddReference("DHI.Mike1D.Generic")
from DHI.Mike1D.Generic import Connection  # noqa


def read(file_path):
    res1d = Res1D(file_path)
    df = pd.DataFrame(index=res1d.time_index)
    for data_set in res1d.data.DataSets:
        for data_item in data_set.DataItems:
            data_set_name = str(data_set.ToString())
            for values, col_name in Res1D.get_values(data_item, data_set_name):
                df[col_name] = values

    return df


class Res1D:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self._time_index = None
        self._load_file()

    def _load_file(self):
        if not os.path.exists(self.file_path):
            raise FileExistsError(f"File {self.file_path} does not exist.")

        self.data = ResultData()
        self.data.Connection = Connection.Create(self.file_path)
        self.data.Load()  # TODO add diagnostics
        self.query = ResultDataQuery(self.data)

    @staticmethod
    def get_values(data_item, data_set_name):
        if data_item.IndexList is None:
            yield data_item.CreateTimeSeriesData(0), data_set_name
        else:
            for i in range(0, data_item.NumberOfElements):
                col_name_i = ':'.join([data_set_name, str(i)])
                yield data_item.CreateTimeSeriesData(i), col_name_i

    @property
    def time_index(self):
        """panda.DatetimeIndex of the time index"""
        if self._time_index:
            return self._time_index

        time_stamps = [from_dotnet_datetime(t) for t in self.data.TimesList]
        self._time_index = pd.DatetimeIndex(time_stamps)
        return self._time_index

    @property
    def quantities(self):
        return [q.Id for q in self.data.Quantities]
