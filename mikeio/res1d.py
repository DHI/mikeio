
def read(res1DFile, extractionPoints):
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

    if os.path.isfile(res1DFile) is False:
        print("ERROR, File Not Found: " + res1DFile)

    # Create a ResultData object and read the data file.
    rd = ResultData()
    rd.Connection = Connection.Create(res1DFile)
    rd.Load()

    reachNums = []
    dataItemTypes = []
    indices = []

    tol = 0.1

    # Find the Item
    for ep in extractionPoints:
        item = -1
        reachNumber = -1
        idx = -1
        for i in range(0, rd.Reaches.Count):
            if rd.Reaches.get_Item(i).Name.lower().strip() == ep.BranchName.lower().strip():

                reach = rd.Reaches.get_Item(i)
                for j in range(0, reach.GridPoints.Count):
                    # print(str(j))
                    if abs(float(reach.GridPoints.get_Item(j).Chainage) - ep.Chainage) < tol:
                        if 'waterlevel' in ep.VariableType.lower().strip().replace(" ", ""):
                            idx = int(j / 2)
                        elif 'discharge' in ep.VariableType.lower().strip().replace(" ", ""):

                            idx = int((j - 1) / 2)
                        elif 'pollutant' in ep.VariableType.lower().strip().replace(" ", ""):
                            idx = int((j - 1) / 2)
                        else:
                            print('ERROR. Variable Type must be either Water Level, Discharge, or Pollutant')
                        reachNumber = i
                        break
                        break
                        break

        for i in range(0, rd.get_Quantities().Count):
            if ep.VariableType.lower().strip().replace(" ", "") == rd.get_Quantities().get_Item(
                    i).Description.lower().strip().replace(" ", ""):
                item = i
                break

        indices.append(idx)
        reachNums.append(reachNumber)
        dataItemTypes.append(item)

    if -1 in reachNums:
        print('ERROR. Reach Not Found')
        quit()
    if -1 in dataItemTypes:
        print('ERROR. Item Not Found')
        quit()
    if -1 in indices:
        print('ERROR. Chainage Not Found')
        quit()

        # Get the Data
    df = pd.DataFrame()
    for i in range(0, len(indices)):
        d = rd.Reaches.get_Item(reachNums[i]).get_DataItems().get_Item(dataItemTypes[i]).CreateTimeSeriesData(
            indices[i])
        name = extractionPoints[i].VariableType + ' ' + str(extractionPoints[i].BranchName) + ' ' + str(
            extractionPoints[i].Chainage)
        d = pd.Series(list(d))
        d = d.rename(name)
        df[name] = d

    # Get the Times
    times = []
    for i in range(0, rd.TimesList.Count):
        it = rd.TimesList.get_Item(i)
        t = pd.Timestamp(
            datetime.datetime(it.get_Year(), it.get_Month(), it.get_Day(), it.get_Hour(), it.get_Minute(),
                              it.get_Second()))
        times.append(t)

    df.index = pd.DatetimeIndex(times)

    rd.Dispose()

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
