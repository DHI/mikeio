import clr
import os.path
import pandas as pd

clr.AddReference("DHI.Mike1D.ResultDataAccess")
from DHI.Mike1D.ResultDataAccess import ResultData

clr.AddReference("DHI.Mike1D.Generic")
from DHI.Mike1D.Generic import Connection

clr.AddReference("System")


class Res1D:

    @staticmethod
    def __read(file_path):
        """
        Read the res1d file
        """
        if not os.path.exists(file_path):
            raise FileExistsError(f"File does not exist {file_path}")

        file = ResultData()
        file.Connection = Connection.Create(file_path)
        file.Load()
        return file

    @staticmethod
    def _get_time(file):
        for i in range(0, file.TimesList.Count):
            t = file.TimesList.get_Item(i)
            yield pd.Timestamp(year=t.get_Year(),
                               month=t.get_Month(),
                               day=t.get_Day(),
                               hour=t.get_Hour(),
                               minute=t.get_Minute(),
                               second=t.get_Second())

    @staticmethod
    def _get_values(dataItemTypes, file, indices, queries, reachNums):
        df = pd.DataFrame()
        for i in range(0, len(indices)):
            d = (file.Reaches.get_Item(reachNums[i])
                 .get_DataItems()
                 .get_Item(dataItemTypes[i])
                 .CreateTimeSeriesData(indices[i]))
            name = str(queries[i])
            d = pd.Series(list(d), name=name)
            df[name] = d
        return df

    def _get_data(self, file, queries, dataItemTypes, indices, reachNums):
        data = self._get_values(dataItemTypes, file, indices, queries, reachNums)
        time = self._get_time(file)
        data.index = pd.DatetimeIndex(time)
        return data

    @staticmethod
    def format_string(s):
        return s.lower().strip().replace(" ", "")

    def find_items(self, file, queries, chainage_tolerance=0.1):
        reachNums = []
        dataItemTypes = []
        indices = []

        # Find the Item
        for query in queries:
            item = -1
            reachNumber = -1
            idx = -1
            for i in range(0, file.Reaches.Count):
                if (file.Reaches.get_Item(i).Name.lower().strip()
                        == query.BranchName.lower().strip()):

                    reach = file.Reaches.get_Item(i)
                    for j in range(0, reach.GridPoints.Count):
                        chainage_diff = float(reach.GridPoints.get_Item(j).Chainage) - query.Chainage
                        is_correct_chainage = abs(chainage_diff) < chainage_tolerance
                        if is_correct_chainage:
                            if "waterlevel" in self.format_string(query.VariableType):
                                idx = int(j / 2)
                            elif "discharge" in self.format_string(query.VariableType):
                                idx = int((j - 1) / 2)
                            elif "pollutant" in self.format_string(query.VariableType):
                                idx = int((j - 1) / 2)
                            else:
                                raise Exception("VariableType must be either Water Level, Discharge, or Pollutant.")
                            reachNumber = i
                            break

            for i in range(0, file.get_Quantities().Count):
                if self.format_string(query.VariableType) == self.format_string(
                        file.get_Quantities().get_Item(i).Description):
                    item = i
                    break

            indices.append(idx)
            reachNums.append(reachNumber)
            dataItemTypes.append(item)

            if -1 in reachNums:
                raise Exception("Reach Not Found")
            if -1 in dataItemTypes:
                raise Exception("Item Not Found")
            if -1 in indices:
                raise Exception("Chainage Not Found")

        return dataItemTypes, indices, reachNums

    def read(self, file_path, queries):

        file = self.__read(file_path)
        dataItemTypes, indices, reachNums = self.find_items(file, queries)
        df = self._get_data(file, queries, dataItemTypes, indices, reachNums)
        file.Dispose()
        return df


class QueryData:
    """A query object that declares what data should be
    extracted from a .res1d file.
    
    Parameters
    ----------
    VariableType: str
        Either 'WaterLevel', 'Discharge' or 'Pollutant'
    BranchName: str, optional
        Branch name, consider all the branches if None
    Chainage: float, optional
        Chainage, considers all the chainages if None
    
    Examples
    --------
    `QueryData('WaterLevel', 'branch1', 10)` is a valid query.
    `QueryData('WaterLevel', 'branch1')` requests all the WaterLevel points
    of `branch1`.
    `QueryData('Discharge')` requests all the Discharge points of the model.
    """

    def __init__(self, VariableType, BranchName=None, Chainage=None):
        self._VariableType = VariableType
        self._BranchName = BranchName
        self._Chainage = Chainage
        self._validate()

    def _validate(self):
        vt = self.VariableType
        bn = self.BranchName
        c = self.Chainage
        if not isinstance(vt, str):
            raise TypeError("VariableType must be a string.")
        if not vt in ["WaterLevel", "Discharge", "Pollutant"]:
            raise ValueError(
                f"Bad VariableType {vt} entered. "
                "It must be either 'WaterLevel', 'Discharge' or 'Pollutant'."
                )
        if bn is not None and not isinstance(bn, str):
            raise TypeError("BranchName must be either None or a string.")
        if c is not None and not isinstance(c, (int, float)):
            raise TypeError("Chainage must be either None or a number.")
        if bn is None and c is not None:
            raise ValueError("Chainage cannot be set if BranchName is None.")
    
    @property
    def VariableType(self):
        return self._VariableType

    @property
    def BranchName(self):
        return self._BranchName

    @property
    def Chainage(self):
        return self._Chainage

    def __repr__(self):
        return (
            f"QueryData(VariableType='{self.VariableType}', "
            f"BranchName='{self.BranchName}', "
            f"Chainage={self.Chainage})"
        )

    def __iter__(self):
        yield self.VariableType
        yield self.BranchName
        yield self.Chainage
