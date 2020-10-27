import os.path
import clr
import pandas as pd
import numpy as np

from mikeio.dotnet import from_dotnet_datetime, to_numpy, to_dotnet_datetime

from System import Enum, DateTime
from DHI.Mike1D.ResultDataAccess import ResultData, ResultDataQuery
from DHI.Mike1D.Generic import Connection, Diagnostics, PredefinedQuantity


def mike1d_quantities():
    """
    Returns all predefined Mike1D quantities.
    Useful for knowing what quantity string to query.
    """
    return [quantity for quantity in Enum.GetNames(clr.GetClrType(PredefinedQuantity))]


class QueryData:
    """A query object that declares what data should be
    extracted from a .res1d file.

    Parameters
    ----------
    quantity: str
        Either 'WaterLevel', 'Discharge', 'Pollutant', 'LeftLatLinkOutflow',
        'RightLatLinkOutflow'
    name: str, optional
        Reach or node or catchment name, consider all if None

    Examples
    --------
    `QueryData('WaterLevel', 'reach1', 10)` is a valid query.
    `QueryData('WaterLevel', 'reach1')` requests all the WaterLevel points
    of `reach1`.
    `QueryData('Discharge')` requests all the Discharge points of the file.
    """

    def __init__(self, quantity, name=None, validate=True):
        self._name = name
        self._quantity = quantity

        if validate:
            self._validate()

    def _validate(self):
        if not isinstance(self.quantity, str):
            raise TypeError("quantity must be a string.")

        if not self.quantity in mike1d_quantities():
            raise ValueError(
                f"Undefined quantity {self.quantity}. Allowed quantities are: "
                f"{', '.join(mike1d_quantities())}."
            )
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("name must be either None or a string.")

    @staticmethod
    def from_dotnet_to_python(array):
        return np.fromiter(array, np.float64)

    @property
    def quantity(self):
        return self._quantity

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return ':'.join([self._quantity, self._name])


class QueryDataReach(QueryData):

    def __init__(self, quantity, name=None, chainage=None, validate=True):
        super().__init__(quantity, name, validate=False)
        self._chainage = chainage

        if validate:
            self._validate()

    def _validate(self):
        super()._validate()

        if self.chainage is not None and not isinstance(self.chainage, (int, float)):
            raise TypeError("chainage must be either None or a number.")
        if self.name is None and self.chainage is not None:
            raise ValueError("chainage cannot be set if name is None.")

    def get_values(self, res1d):
        values = res1d.query.GetReachValues(self._name, self._chainage, self._quantity)
        return self.from_dotnet_to_python(values)

    @property
    def chainage(self):
        return self._chainage

    def __repr__(self):
        return ':'.join([self._quantity, self._name, str(self._chainage)])


class QueryDataNode(QueryData):

    def __init__(self, quantity, name=None, validate=True):
        super().__init__(quantity, name, validate)

    def get_values(self, res1d):
        values = res1d.query.GetNodeValues(self._name, self._quantity)
        return self.from_dotnet_to_python(values)


class Res1D:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self._time_index = None
        self._start_time = None
        self._end_time = None
        self._load_file()

    def _load_file(self):
        if not os.path.exists(self.file_path):
            raise FileExistsError(f"File {self.file_path} does not exist.")

        self._data = ResultData()
        self._data.Connection = Connection.Create(self.file_path)
        self._data.Load(Diagnostics())
        self._query = ResultDataQuery(self._data)

    def read(self, queries):
        df = pd.DataFrame(index=self.time_index)
        for query in queries:
            df[str(query)] = query.get_values(self)

        return df

    @staticmethod
    def read_to_dataframe(file_path, queries=None):
        """ Read all or queried data in res1d file to a pandas DataFrame."""

        res1d = Res1D(file_path)

        if queries is not None:
            queries = queries if isinstance(queries, list) else [queries]
            return res1d.read(queries)
        # TODO else: create_all_queries(res1d)

        df = pd.DataFrame(index=res1d.time_index)
        for data_set in res1d.data.DataSets:
            for data_item in data_set.DataItems:
                for values, col_name in Res1D.get_values(data_set, data_item):
                    df[col_name] = values

        return df

    @staticmethod
    def get_values(data_set, data_item, col_name_delimiter=':', put_chainage_in_col_name=True):
        """ Get all time series values in given data_item. """
        name = data_set.Name if hasattr(data_set, 'Name') else data_set.Id
        if data_item.IndexList is None or data_item.NumberOfElements == 1:
            col_name = col_name_delimiter.join([data_item.Quantity.Id, name])
            yield data_item.CreateTimeSeriesData(0), col_name
        else:
            chainages = data_set.GetChainages(data_item)
            for i in range(0, data_item.NumberOfElements):
                if put_chainage_in_col_name:
                    postfix = f"{chainages[i]:g}"
                else:
                    postfix = str(i)

                col_name_i = col_name_delimiter.join([data_item.Quantity.Id, name, postfix])
                yield data_item.CreateTimeSeriesData(i), col_name_i

    @property
    def time_index(self):
        """panda.DatetimeIndex of the time index."""
        if self._time_index is not None:
            return self._time_index

        time_stamps = [from_dotnet_datetime(t) for t in self.data.TimesList]
        self._time_index = pd.DatetimeIndex(time_stamps)
        return self._time_index

    @property
    def start_time(self):
        if self._start_time is not None:
            return self._start_time

        return from_dotnet_datetime(self.data.StartTime)

    @property
    def end_time(self):
        if self._end_time is not None:
            return self._end_time

        return from_dotnet_datetime(self.data.EndTime)

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

    def get_node_values(self, node_id, quantity):
        return to_numpy(self.query.GetNodeValues(node_id, quantity))

    def get_reach_values(self, reach_name, chainage, quantity):
        return to_numpy(self.query.GetReachValues(reach_name, chainage, quantity))

    def get_reach_value(self, reach_name, chainage, quantity, time):
        time_dotnet = time if isinstance(time, DateTime) else to_dotnet_datetime(time)
        return self.query.GetReachValue(reach_name, chainage, quantity, time_dotnet)

    def get_reach_start_values(self, reach_name, quantity):
        return to_numpy(self.query.GetReachStartValues(reach_name, quantity))

    def get_reach_end_values(self, reach_name, quantity):
        return to_numpy(self.query.GetReachEndValues(reach_name, quantity))

    def get_reach_sum_values(self, reach_name, quantity):
        return to_numpy(self.query.GetReachSumValues(reach_name, quantity))
