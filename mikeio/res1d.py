from collections import defaultdict, namedtuple
from contextlib import contextmanager
import functools

import clr
import os.path
import pandas as pd

clr.AddReference("DHI.Mike1D.ResultDataAccess")
from DHI.Mike1D.ResultDataAccess import ResultData  # noqa

clr.AddReference("DHI.Mike1D.Generic")
from DHI.Mike1D.Generic import Connection  # noqa

clr.AddReference("System")

# Data types handled by the read function.
DATA_TYPES_HANDLED_IN_QUERIES = [
    "WaterLevel",
    "Discharge",
    "Pollutant",
    "LeftLatLinkOutflow",
    "RightLatLinkOutflow",
]

class BaseRes1DError(Exception):
    """Base class for Red1D errors."""


class DataNotFoundInFile(BaseRes1DError):
    """Data not found in file."""


class FileNotOpenedError(BaseRes1DError):
    """Accessing data from a file that is not opened."""


def read(file_path, queries):
    """Read the requested data from the res1d file and
    return a Pandas DataFrame.
    
    Parameters
    ----------
    file_path: str
        full path and file name to the res1d file.
    queries: a single query or a list of queries
        `QueryData` objects that define the requested data.
    Returns
    -------
    pd.DataFrame
    """
    queries = queries if isinstance(queries, list) else [queries]
    with open(file_path) as res1d:
        return res1d.read(queries)


def open(file_path):
    """Open a res1d file as a Res1D object that has convenient methods
    to extract specific data from the file. It is recommended to use it
    as a context manager.

    Parameters
    ----------
    file_path: str
        full path and file name to the res1d file.
    
    Returns
    -------
    Res1D
    
    Examples
    --------
    >>> with open("file.res1d") as r1d:
    >>>     print(r1d.data_types)
    ['WaterLevel', 'Discharge']
    """
    return Res1D(file_path)


def _not_closed(prop):
    @functools.wraps(prop)
    def wrapper(self, *args, **kwargs):
        if self._closed:
            raise FileNotOpenedError(
                "The res1d file should be opened first to access to its "
                f"{prop.__name__} property."
            )
        return prop(self, *args, **kwargs)
    return wrapper


class Res1D:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.file = None
        self._closed = True
        self._time_index = None
        self._data_types = None
        self._reach_names = None
        self.__reaches = None
        # Load the file on initialization
        self._load_file()

    def _load_file(self):
        """Load the file."""
        if not os.path.exists(self.file_path):
            raise FileExistsError(f"File {self.file_path} does not exist.")
        self.file = ResultData()
        self.file.Connection = Connection.Create(self.file_path)
        self.file.Load()
        self._closed = False

    def close(self):
        """Close the file handle."""
        self.file.Dispose()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *excinfo):
        self.close()

    @property
    @_not_closed
    def data_types(self):
        """List of the data types"""
        if self._data_types:
            return self._data_types
        return [q.Id for q in self.file.get_Quantities()]

    @property
    def _reaches(self):
        if self.__reaches:
            return self.__reaches
        return list(self.file.Reaches)

    @property
    @_not_closed
    def reach_names(self):
        """A list of the reach names"""
        if self._reach_names:
            return self._reach_names
        return [reach.Name for reach in self._reaches]

    @staticmethod
    def _chainages(reach, data_type_idx):
        """Generates chainages given a reach object and a data_type_idx"""
        data_item = list(reach.DataItems)[data_type_idx]
        index_list = list(data_item.IndexList)
        gridpoints = list(reach.GridPoints)
        gridpoints_filtered = [gridpoints[i] for i in index_list]
        for gp in gridpoints_filtered:
            yield float(gp.Chainage)

    @staticmethod
    def _data_types_reach(reach):
        """A list of the data types IDs contained in a reach."""
        return [di.get_Quantity().Id for di in list(reach.get_DataItems())]

    @property
    @_not_closed
    def time_index(self):
        """panda.DatetimeIndex of the time index"""
        if self._time_index:
            return self._time_index
        time_stamps = []
        for t in self.file.TimesList:
            time_stamps.append(
                pd.Timestamp(
                    year=t.get_Year(),
                    month=t.get_Month(),
                    day=t.get_Day(),
                    hour=t.get_Hour(),
                    minute=t.get_Minute(),
                    second=t.get_Second()
                )
            )
        self._time_index = pd.DatetimeIndex(time_stamps)
        return self._time_index

    def _get_values(self, points):
        df = pd.DataFrame()
        p = zip(points["variable"], points["reach"], points["chainage"])
        for variable_type, reach, chainage in p:
            d = (self.file.Reaches.get_Item(reach.index)
                 .get_DataItems()
                 .get_Item(variable_type.index)
                 .CreateTimeSeriesData(chainage.index))
            name = f"{variable_type.value} {reach.value} {chainage.value:.3f}"
            d = pd.Series(list(d), name=name)
            df[name] = d
        return df

    def _get_data(self, points):
        df = self._get_values(points)
        df.index = self.time_index
        return df

    def _validate_queries(self, queries, chainage_tolerance=0.1):
        """Check whether the queries point to existing data in the file."""
        for q in queries:
            # Raise an error if the data type is not found globally
            if q.variable_type not in self.data_types:
                raise DataNotFoundInFile(
                    f"Data type '{q.variable_type}' was not found.")
            if q.reach_name is not None:
                if q.reach_name not in self.reach_names:
                    raise DataNotFoundInFile(
                        f"Reach '{q.reach_name}' was not found.")
            if q.chainage is not None:
                found_chainage = False
                for reach in self._reaches:
                    if found_chainage:
                        break
                    # Look for the targeted reach
                    if q.reach_name != reach.Name:
                        continue
                    # Raise an error if the data type isn't found in this reach
                    data_types_in_reach = self._data_types_reach(reach)
                    if q.variable_type not in data_types_in_reach:
                        raise DataNotFoundInFile(
                            f"Data type '{q.variable_type}' was not found.")
                    data_type_idx = data_types_in_reach.index(q.variable_type)
                    for chainage in self._chainages(reach, data_type_idx):
                        # Look for the targeted chainage
                        chainage_diff = chainage - q.chainage
                        if abs(chainage_diff) < chainage_tolerance:
                            found_chainage = True
                            break
                if not found_chainage:
                    raise DataNotFoundInFile(
                        f"Chainage {q.chainage} was not found.")

    def _build_queries(self, queries):
        """"
        A query can be in an undefined state if reach_name and/or chainage
        isn't set. This function takes care of building lists of queries
        for these cases. Chainages are rounded to three decimal places.

        >>> self._build_queries([QueryData("WaterLevel", "reach1")])
        [
            QueryData("WaterLevel", "reach1", 0),
            QueryData("WaterLevel", "reach1", 10)
        ]
        """
        built_queries = []
        for q in queries:
            # e.g. QueryData("WaterLevel", "reach1", 1)
            if q.reach_name and q.chainage:
                built_queries.append(q)
                continue
            # e.g QueryData("WaterLevel", "reach1") or QueryData("WaterLevel")
            q_variable_type = q.variable_type
            q_reach_name = q.reach_name
            for reach, reach_name in zip(self._reaches, self.reach_names):
                if q_reach_name is not None:  # When reach_name is set.
                    if reach_name != q_reach_name:
                        continue
                data_types_in_reach = self._data_types_reach(reach)
                if q.variable_type not in data_types_in_reach:
                    continue
                data_type_idx = data_types_in_reach.index(q.variable_type)
                for curr_chain in self._chainages(reach, data_type_idx):
                    if q_variable_type in DATA_TYPES_HANDLED_IN_QUERIES:
                        chainage = curr_chain
                    else:
                        continue

                    q = QueryData(
                        q_variable_type, reach_name, round(chainage, 3)
                    )
                    built_queries.append(q)
        return built_queries

    def _find_points(self, queries, chainage_tolerance=0.1):
        """From a list of queries returns a dictionary with the required
        information for each requested point to extract its time series
        later on."""

        PointInfo = namedtuple('PointInfo', ['index', 'value'])

        found_points = defaultdict(list)
        # Find the point given its variable type, reach, and chainage
        for q in queries:
            for reach_idx, curr_reach in enumerate(self._reaches):
                # Look for the targeted reach
                if q.reach_name != curr_reach.Name:
                    continue
                reach = PointInfo(reach_idx, q.reach_name)
                for data_type_idx, data_type in enumerate(self.data_types):
                    if q.variable_type.lower() == data_type.lower():
                        break
                data_type_info = PointInfo(data_type_idx, q.variable_type)
                for idx, curr_chain in enumerate(self._chainages(curr_reach, data_type_idx)):
                    # Look for the targeted chainage
                    chainage_diff = curr_chain - q.chainage
                    is_chainage = abs(chainage_diff) < chainage_tolerance
                    if not is_chainage:
                        continue
                    # idx is the index in the item data, not in the
                    # gridpoints data.
                    chainage = PointInfo(idx, q.chainage)
                    found_points["chainage"].append(chainage)
                    found_points["variable"].append(data_type_info)
                    found_points["reach"].append(reach)
                    break  # Break at the first chainage found.

        return dict(found_points)

    @_not_closed
    def read(self, queries):
        """Read the requested data from the res1d file and
        return a Pandas DataFrame.

        Parameters
        ----------
        queries: list
            `QueryData` objects that define the requested data.
        Returns
        -------
        pd.DataFrame
        """
        self._validate_queries(queries)
        built_queries = self._build_queries(queries)
        found_points = self._find_points(built_queries)
        df = self._get_data(found_points)
        return df


class QueryData:
    """A query object that declares what data should be
    extracted from a .res1d file.
    
    Parameters
    ----------
    variable_type: str
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

    allowed_data_types = DATA_TYPES_HANDLED_IN_QUERIES

    def __init__(self, variable_type, reach_name=None, chainage=None):
        self._variable_type = variable_type
        self._reach_name = reach_name
        self._chainage = chainage
        self._validate()

    def _validate(self):
        vt = self.variable_type
        rn = self.reach_name
        c = self.chainage
        if not isinstance(vt, str):
            raise TypeError("variable_type must be a string.")
        if not vt in self.allowed_data_types:
            raise ValueError(
                f"Undefined variable_type {vt}. Allowed types are: "
                f"{', '.join(self.allowed_data_types)}."
            )
        if rn is not None and not isinstance(rn, str):
            raise TypeError("reach_name must be either None or a string.")
        if c is not None and not isinstance(c, (int, float)):
            raise TypeError("chainage must be either None or a number.")
        if rn is None and c is not None:
            raise ValueError("chainage cannot be set if reach_name is None.")

    @property
    def variable_type(self):
        return self._variable_type

    @property
    def reach_name(self):
        return self._reach_name

    @property
    def chainage(self):
        return self._chainage

    def __repr__(self):
        return (
            f"QueryData(variable_type='{self.variable_type}', "
            f"reach_name='{self.reach_name}', "
            f"chainage={self.chainage})"
        )
