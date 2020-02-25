import clr
import pandas as pd
import datetime
import numpy as np
import os.path

clr.AddReference("DHI.Mike1D.ResultDataAccess")
from DHI.Mike1D.ResultDataAccess import ResultData

clr.AddReference("DHI.Mike1D.Generic")
from DHI.Mike1D.Generic import Connection

clr.AddReference("System")


class res1d:

    # chainage_tolerance = 0.1

    def __read(self, file_path):
        """Read data from the res1d file
        """
        if not os.path.exists(file_path):
            raise("File does not exist %s", file_path)

        file = ResultData()
        file.Connection = Connection.Create(file_path)
        file.Load()
        return file

    def get_time(self, file):
        times = []
        for i in range(0, file.TimesList.Count):
            it = file.TimesList.get_Item(i)
            t = pd.Timestamp(
                datetime.datetime(
                    it.get_Year(),
                    it.get_Month(),
                    it.get_Day(),
                    it.get_Hour(),
                    it.get_Minute(),
                    it.get_Second(),
                )
            )
            times.append(t)
        return times

    def get_values(self, dataItemTypes, file, indices, queries, reachNums):
        df = pd.DataFrame()
        for i in range(0, len(indices)):
            d = (
                file.Reaches.get_Item(reachNums[i])
                    .get_DataItems()
                    .get_Item(dataItemTypes[i])
                    .CreateTimeSeriesData(indices[i])
            )
            name = (
                    queries[i].VariableType
                    + " "
                    + str(queries[i].BranchName)
                    + " "
                    + str(queries[i].Chainage)
            )
            d = pd.Series(list(d))
            d = d.rename(name)
            df[name] = d
        return df

    def get_data(self, dataItemTypes, file, indices, queries, reachNums):
        df = self.get_values(dataItemTypes, file, indices, queries, reachNums)

        times = self.get_time(file)
        df.index = pd.DatetimeIndex(times)
        return df

    @staticmethod
    def format_string(s):
        return s.lower().strip().replace(" ", "")

    def parse_query(self, query):
        return query.VariableType.lower().strip().replace(" ", "")

    def read(self, file_path, queries):

        file = self.__read(file_path)

        reachNums = []
        dataItemTypes = []
        indices = []

        tol = 0.1

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
                        isCorrectChainage = abs(float(reach.GridPoints.get_Item(j).Chainage) - query.Chainage) < tol
                        if (isCorrectChainage):
                            if "waterlevel" in self.format_string(query.VariableType):
                                idx = int(j / 2)
                            elif "discharge" in self.format_string(query.VariableType):
                                idx = int((j - 1) / 2)
                            elif "pollutant" in self.format_string(query.VariableType):
                                idx = int((j - 1) / 2)
                            else:
                                raise("VariableType must be either Water Level, Discharge, or Pollutant.")
                            reachNumber = i
                            break
                            break
                            break

            for i in range(0, file.get_Quantities().Count):
                if self.format_string(query.VariableType) == self.format_string(file.get_Quantities().get_Item(i).Description):
                    item = i
                    break

            indices.append(idx)
            reachNums.append(reachNumber)
            dataItemTypes.append(item)

        if -1 in reachNums:
            raise("Reach Not Found")
        if -1 in dataItemTypes:
            raise("Item Not Found")
        if -1 in indices:
            raise("Chainage Not Found")

        df = self.get_data(dataItemTypes, file, indices, queries, reachNums)

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
