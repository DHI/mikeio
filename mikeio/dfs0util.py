import numpy as np
from mikecore.DfsFile import DfsItemData


class Dfs0Util:
  
    @staticmethod
    def ReadDfs0DataDouble(dfs0File):
        
        itemCount = len(dfs0File.ItemInfo)
        timestepCount = dfs0File.FileInfo.TimeAxis.NumberOfTimeSteps
        res = np.empty(shape=(timestepCount, itemCount+1), dtype=np.float64)

        # Preload a set of item data
        itemDatas = []
        for j in range(itemCount):
            itemDatas.append(dfs0File.CreateEmptyItemData(j+1))
        
        dfs0File.Reset()

        # Check if time axis is really a time axis, or if it is a non-time axis
        # TimeAxisType timeAxisType = dfs0File.FileInfo.TimeAxis.TimeAxisType;
        #timeUnit = dfs0File.FileInfo.TimeAxis.TimeUnit
        # isTimeUnit = EUMWrapper.eumUnitsEqv((int) eumUnit.eumUsec, timeUnit)
        # isTimeUnit = True

        for i in range(timestepCount):
            for j in range(itemCount):
                itemData = itemDatas[j]
                dfs0File.ReadItemTimeStep(itemData, i)
                # First column is time, remaining colums are data
                if (j == 0):
#                    if isTimeUnit:
#                        res[i, 0] = itemData.TimeInSeconds(dfs0File.FileInfo.TimeAxis)
#                    else:  # not a time-unit, just return the value
                    res[i, 0] = itemData.Time

                    res[i, j+1] = itemData.Data[0]
            
        return res
    
    #@staticmethod  
    #def WriteDfs0DataDouble(dfs0File, times, data):
    
    #  itemCount = dfs0File.ItemInfo.Count;

    #  if (times.Length != data.GetLength(0)):
    #    throw new ArgumentException("Number of time steps does not match number of data rows");

    #  if (itemCount != data.GetLength(1))
    #    throw new ArgumentException("Number of items does not match number of data columns");

    #  bool[] isFloatItem = new bool[itemCount];
    #  for (int j = 0; j < itemCount; j++)
    #  {
    #    isFloatItem[j] = dfs0File.ItemInfo[j].DataType == DfsSimpleType.Float;
    #  }

    #  float[] fdata = new float[1];
    #  double[] ddata = new double[1];

    #  dfs0File.Reset();

    #  for (int i = 0; i < times.Length; i++)
    #  {
    #    for (int j = 0; j < itemCount; j++)
    #    {
    #      if (isFloatItem[j])
    #      {
    #        fdata[0] = (float)data[i, j];
    #        dfs0File.WriteItemTimeStepNext(times[i], fdata);
    #      }
    #      else
    #      {
    #        ddata[0] = data[i, j];
    #        dfs0File.WriteItemTimeStepNext(times[i], ddata);
