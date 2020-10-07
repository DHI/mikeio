import os.path

import clr
import pandas as pd

from mikeio.dotnet import from_dotnet_datetime

clr.AddReference("System")
from System import Enum

clr.AddReference("DHI.Mike1D.ResultDataAccess")
from DHI.Mike1D.ResultDataAccess import ResultData, ResultDataQuery

clr.AddReference("DHI.Mike1D.Generic")
from DHI.Mike1D.Generic import Connection, Diagnostics, PredefinedQuantity


def mike1d_quantities():
    """ List all possible Mike1D quantities."""
    return [quantity for quantity in Enum.GetNames(clr.GetClrType(PredefinedQuantity))]


def read(file_path):
    """ Read all data in res1d file to a pandas DataFrame."""
    res1d = Res1D(file_path)
    df = pd.DataFrame(index=res1d.time_index)
    for data_set in res1d.data.DataSets:
        for data_item in data_set.DataItems:
            # data_set.Id + data_item.get_Quantity().Id + chainage 
            for values, col_name in Res1D.get_values(data_item, data_set.Id):
                df[col_name] = values

    return df


class QueryData:
    """A query object that declares what data should be
    extracted from a .res1d file.

    Parameters
    ----------
    quantity: str
        Either 'WaterLevel', 'Discharge', 'Pollutant', 'LeftLatLinkOutflow',
        'RightLatLinkOutflow'
    reach_name: str, optional
        Reach name, consider all the reaches if None
    chainage: float, optional
        chainage, considers all the chainages if None

    Examples
    --------
    `QueryData('WaterLevel', 'reach1', 10)` is a valid query.
    `QueryData('WaterLevel', 'reach1')` requests all the WaterLevel points
    of `reach1`.
    `QueryData('Discharge')` requests all the Discharge points of the file.
    """

    def __init__(self, quantity, reach_name=None, chainage=None, validate=True):
        self._quantity = quantity
        self._reach_name = reach_name
        self._chainage = chainage

        if validate:
            self._validate()

    def _validate(self):
        if not isinstance(self.quantity, str):
            raise TypeError("variable_type must be a string.")

        if not self.quantity in mike1d_quantities():
            raise ValueError(
                f"Undefined quantity {self.quantity}. Allowed quantities are: "
                f"{', '.join(mike1d_quantities())}."
            )
        if self.reach_name is not None and not isinstance(self.reach_name, str):
            raise TypeError("reach_name must be either None or a string.")
        if self.chainage is not None and not isinstance(self.chainage, (int, float)):
            raise TypeError("chainage must be either None or a number.")
        if self.reach_name is None and self.chainage is not None:
            raise ValueError("chainage cannot be set if reach_name is None.")

    @property
    def quantity(self):
        return self._quantity

    @property
    def reach_name(self):
        return self._reach_name

    @property
    def chainage(self):
        return self._chainage

    def __repr__(self):
        return (
            f"QueryData(quantity='{self.quantity}', "
            f"reach_name='{self.reach_name}', "
            f"chainage={self.chainage})"
        )


class Res1D:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self._time_index = None
        self._load_file()

    def _load_file(self):
        if not os.path.exists(self.file_path):
            raise FileExistsError(f"File {self.file_path} does not exist.")

        self._data = ResultData()
        self._data.Connection = Connection.Create(self.file_path)
        self._data.Load(Diagnostics())
        self._query = ResultDataQuery(self._data)

    @staticmethod
    def get_values(data_item, data_set_name):
        """ Get all time series values in given data_item."""
        if data_item.IndexList is None or data_item.NumberOfElements == 1:
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
        """Quantities in res1d file."""
        return [quantity.Id for quantity in self._data.Quantities]

    @property
    def query(self):
        """
        Object to use for querying the loaded res1d data.
        Returns a C# ResultDataQuery object that has the following methods
        //
        // Summary:
        //     Find element in dataItem that is closest to chainage
        public int FindClosestElement(IRes1DReach reach, double chainage, IDataItem dataItem);
        //
        // Summary:
        //     Find data item in dataSet which quantity matches the given quantityId.
        //     Returns null if none found
        public IDataItem FindDataItem(IRes1DDataSet dataSet, string quantityId);
        //
        // Summary:
        //     Get time series values for quantityId of catchment with id catchmentId. If catchment
        //     or quantity could not be found, null is returned.
        public float[] GetCatchmentValues(string catchmentId, string quantityId);
        //
        // Summary:
        //     Get result file datetimes.
        public DateTime[] GetDateTimes();
        //
        // Summary:
        //     Get result file datetimes as strings.
        public string[] GetDateTimesAsStrings(string format = "u");
        //
        // Summary:
        //     Get time series values for the node with the id nodeId and quantity with id quantityId.
        //     If node or quantity could not be found, null is returned.
        public float[] GetNodeValues(string nodeId, string quantityId);
        //
        // Summary:
        //     Get time series values at the end of of the reach with name reachName and quantity
        //     with id quantityId. If reach or quantity could not be found, null is returned.
        public float[] GetReachEndValues(string reachName, string quantityId);
        //
        // Summary:
        //     Get time series values at the start of of the reach with name reachName and quantity
        //     with id quantityId. If reach or quantity could not be found, null is returned.
        public float[] GetReachStartValues(string reachName, string quantityId);
        //
        // Summary:
        //     Get time series values summing up quantity for all grid points in reach. This
        //     is useful for quantities like water volumes.
        public float[] GetReachSumValues(string reachName, string quantityId);
        //
        // Summary:
        //     Get value at the element that is closest to chainage in reach with name reachName
        //     and quantity with id quantityId, at time time, interpolated if required. If reach
        //     or quantity could not be found, null is returned.
        public float GetReachValue(string reachName, double chainage, string quantityId, DateTime time);
        //
        // Summary:
        //     Get time series values at the element that is closest to chainage in reach with
        //     name reachName and quantity with id quantityId. If reach or quantity could not
        //     be found, null is returned.
        public float[] GetReachValues(string reachName, double chainage, string quantityId);
        """
        return self._query

    @property
    def data(self):
        """
        Object with the loaded res1d data.
        Returns a C# ResultData object that has the following methods:
        // Summary:
        //     Data coverage start
        public DateTime StartTime { get; set; }
        //
        // Summary:
        //     Data coverage end
        public DateTime EndTime { get; set; }
        //
        // Summary:
        //     Number of time steps
        public int NumberOfTimeSteps { get; }
        //
        // Summary:
        //     Time axis for the data.
        public IListDateTimes TimesList { get; set; }
        public ResultTypes ResultType { get; set; }
        //
        // Summary:
        //     List of the contained quantities. Note: This is a derived property
        public IQuantities Quantities { get; }
        //
        // Summary:
        //     List of the contained quantities. Note: This is a derived property
        public IListstrings StructureTypes { get; }
        //
        // Summary:
        //     Get an iterator that iterates over all data items
        public IEnumerable<IDataItem> DataItems { get; }
        //
        // Summary:
        //     Get an iterator that iterates over all data sets
        public IEnumerable<IRes1DDataSet> DataSets { get; }
        //
        // Summary:
        //     List of nodes
        public IRes1DNodes Nodes { get; set; }
        //
        // Summary:
        //     Unit system of the simulation that produced the result data object.
        //     When creating a result data object and storing: Properties of ResultData objects
        //     (coordinates, bottom levels etc.) must always be set in SI units.
        //     When loading a result data object from storage: The DHI.Mike1D.ResultDataAccess.IResultDataParameters.UnitSystem
        //     and DHI.Mike1D.ResultDataAccess.IResultDataParameters.ConvertGeometry can be
        //     used to change units of data and properties in the Result Data object. This property
        //     will maintain the original value and will not be changed by updating the DHI.Mike1D.ResultDataAccess.IResultDataParameters
        public UnitSystem UnitSystem { get; set; }
        //
        // Summary:
        //     List of branches
        public IRes1DReaches Reaches { get; set; }
        //
        // Summary:
        //     Global data. Valid for entire network
        public IRes1DGlobalData GlobalData { get; set; }
        //
        // Summary:
        //     Static data on the network.
        //     Used for user defined markers from MIKE 11 and Critical Levels in MU.
        public IList<INetworkDataDouble> NetworkDatas { get; }
        public float DeleteValue { get; set; }
        public double SecondsBetweenFileFlush { get; set; }
        //
        // Summary:
        //     Result specification
        public ResultSpecification ResultSpecs { get; set; }
        public LoadStatus LoadStatus { get; }
        //
        // Summary:
        //     List of catchments
        public IRes1DCatchments Catchments { get; set; }
        //
        // Summary:
        //     A WKT string for a spatial reference system.
        public string ProjectionString { get; set; }
        """
        return self._data
