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
            raise ("File does not exist %s", file_path)

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


class ExtractionPoint:
    def VariableType(VariableType):
        """
            Variable Type (eg. WaterLevel or Discharge)
        """
        return VariableType

    def BranchName(BranchName):
        """
            Name of the Branch
        """
        return BranchName

    def Chainage(Chainage):
        """
            Chainage number along branch
        """
        return Chainage

    def __str__(self):
        return f"{self.VariableType} {self.BranchName} {self.Chainage}"
