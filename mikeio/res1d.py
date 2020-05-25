from collections import defaultdict, namedtuple
from contextlib import contextmanager

import clr
import os.path
import pandas as pd

clr.AddReference("DHI.Mike1D.ResultDataAccess")
from DHI.Mike1D.ResultDataAccess import ResultData  # noqa

clr.AddReference("DHI.Mike1D.Generic")
from DHI.Mike1D.Generic import Connection  # noqa

clr.AddReference("System")


class DataNotFoundInFile(Exception):
    """Data not found in file."""


def read(file_path, queries):
    """Read the requested data from the res1d file and
    return a Pandas DataFrame.
    
    Parameters
    ----------
    file_path: str
        full path and file name to the res1d file.
    queries: list
        `QueryData` objects that define the requested data.
    Returns
    -------
    pd.DataFrame
    """
    res1d = Res1D(file_path)
    df = res1d.read(queries)
    return df


def _comp_strings(s1, s2, format=None):
    if not format:
        return s1 == s2
    return format(s1) == format(s2)


class Res1D:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.file = None
        self._time_index = None
        self._data_types = None
        self._reaches = None

    def _load_file(self):
        """Read the res1d file."""
        if not os.path.exists(self.file_path):
            raise FileExistsError(f"File {self.file_path} does not exist.")

        file = ResultData()
        file.Connection = Connection.Create(self.file_path)
        file.Load()
        return file

    def _close(self):
        self.file.Dispose()

    @contextmanager
    def open(self):
        yield self._load_file()
        self._close()

    @property
    def data_types(self):
        if self._data_types:
            return self._data_types
        quantities = self.file.get_Quantities()
        return [quantities.get_Item(i).Id for i in range(0, quantities.Count)]

    @property
    def reaches(self):
        if self._reaches:
            return self._reaches
        reaches = self.file.Reaches
        return [reaches.get_Item(i) for i in range(0, reaches.Count)]

    @staticmethod
    def _chainages(reach):
        for i in range(0, reach.GridPoints.Count):
            yield float(reach.GridPoints.get_Item(i).Chainage)

    @property
    def time_index(self):
        if self._time_index:
            return self._time_index
        time_stamps = []
        for i in range(0, self.file.TimesList.Count):
            t = self.file.TimesList.get_Item(i)
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
            name = f"{variable_type.value} {reach.value} {chainage.value}"
            d = pd.Series(list(d), name=name)
            df[name] = d
        return df

    def _get_data(self, points):
        df = self._get_values(points)
        df.index = self.time_index
        return df

    def _find_points(self, queries, chainage_tolerance=0.1):
        """From a list of queries returns a dictionary with the required
        information for each requested point to extract its time series
        later on."""

        PointInfo = namedtuple('PointInfo', ['index', 'value'])

        found_points = defaultdict(list)
        # Find the point
        for q_variable_type, q_reach_name, q_chain in queries:

            found_data_type = found_reach = found_chainage = False

            for data_type_idx, data_type in enumerate(self.data_types):
                if q_variable_type.lower() == data_type.lower():
                    found_data_type = True
                    break
            if not found_data_type:
                raise DataNotFoundInFile(
                    f"Data type '{q_variable_type}' was not found.")
            data_type_info = PointInfo(data_type_idx, q_variable_type)
            if q_reach_name and q_chain:
                found_points["variable"].append(data_type_info)

            for reach_idx, curr_reach in enumerate(self.reaches):
                # Look for the targeted chainage if set
                if q_reach_name:
                    if not q_reach_name == curr_reach.Name:
                        continue

                found_reach = True
                reach_val = q_reach_name if q_reach_name else curr_reach
                reach = PointInfo(reach_idx, reach_val)
                if q_reach_name and q_chain:
                    found_points["reach"].append(reach)
                for j, curr_chain in enumerate(self._chainages(curr_reach)):
                    # Look for the targeted chainage if set.
                    if q_chain:
                        chainage_diff = curr_chain - q_chain
                        is_chainage = abs(chainage_diff) < chainage_tolerance
                        if not is_chainage:
                            continue

                    if q_variable_type == "WaterLevel" and j % 2 == 0:
                        chainage_idx = int(j / 2)
                    elif q_variable_type == "Discharge" and j % 2 != 0:
                        chainage_idx = int((j - 1) / 2)
                    elif q_variable_type == "Pollutant" and j % 2 != 0:
                        chainage_idx = int((j - 1) / 2)
                    else:
                        continue  # q_chainage is None in that case.

                    found_chainage = True
                    chainage_val = q_chain if q_chain else round(curr_chain, 3)
                    chainage = PointInfo(chainage_idx, chainage_val)
                    found_points["chainage"].append(chainage)
                    if not q_chain:
                        found_points["variable"].append(data_type_info)
                        found_points["reach"].append(reach)
                    else:
                        break  # Break at the first chainage found.

            if not found_reach:
                raise DataNotFoundInFile(
                    f"Reach '{q_reach_name}' was not found.")
            if not found_chainage:
                raise DataNotFoundInFile(
                    f"Chainage {q_chain} was not found.")

        return dict(found_points)

    def read(self, queries):
        with self.open() as self.file:
            found_points = self._find_points(queries)
            df = self._get_data(found_points)
        return df


class QueryData:
    """A query object that declares what data should be
    extracted from a .res1d file.
    
    Parameters
    ----------
    variable_type: str
        Either 'WaterLevel', 'Discharge' or 'Pollutant'
    branch_name: str, optional
        Branch name, consider all the branches if None
    chainage: float, optional
        chainage, considers all the chainages if None
    
    Examples
    --------
    `QueryData('WaterLevel', 'branch1', 10)` is a valid query.
    `QueryData('WaterLevel', 'branch1')` requests all the WaterLevel points
    of `branch1`.
    `QueryData('Discharge')` requests all the Discharge points of the model.
    """

    def __init__(self, variable_type, branch_name=None, chainage=None):
        self._variable_type = variable_type
        self._branch_name = branch_name
        self._chainage = chainage
        self._validate()

    def _validate(self):
        vt = self.variable_type
        bn = self.branch_name
        c = self.chainage
        if not isinstance(vt, str):
            raise TypeError("variable_type must be a string.")
        if not vt in ["WaterLevel", "Discharge", "Pollutant"]:
            raise ValueError(
                f"Bad variable_type {vt} entered. "
                "It must be either 'WaterLevel', 'Discharge' or 'Pollutant'."
            )
        if bn is not None and not isinstance(bn, str):
            raise TypeError("branch_name must be either None or a string.")
        if c is not None and not isinstance(c, (int, float)):
            raise TypeError("chainage must be either None or a number.")
        if bn is None and c is not None:
            raise ValueError("chainage cannot be set if branch_name is None.")

    @property
    def variable_type(self):
        return self._variable_type

    @property
    def branch_name(self):
        return self._branch_name

    @property
    def chainage(self):
        return self._chainage

    def __repr__(self):
        return (
            f"QueryData(variable_type='{self.variable_type}', "
            f"branch_name='{self.branch_name}', "
            f"chainage={self.chainage})"
        )

    def __iter__(self):
        yield self.variable_type
        yield self.branch_name
        yield self.chainage
