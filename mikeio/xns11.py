import clr
import os.path
import pandas as pd

clr.AddReference("DHI.Mike1D.CrossSectionModule")
from DHI.Mike1D.CrossSectionModule import CrossSectionDataFactory

clr.AddReference("DHI.Mike1D.Generic")
from DHI.Mike1D.Generic import Connection, Diagnostics, Location


# clr.AddReference("System")


class Xns11:

    @staticmethod
    def __read(file_path):
        """
        Read the xns11 file
        """
        if not os.path.exists(file_path):
            raise FileExistsError(f"File does not exist {file_path}")

        xns11 = CrossSectionDataFactory()
        file = xns11.Open(Connection.Create(file_path), Diagnostics("Error loading file"))

        return file

    @staticmethod
    def _get_values(file, queries, reaches_name, reaches_topoid, chainages):

        df = pd.DataFrame()

        for i in range(len(reaches_name)):
            location = Location()
            location.ID = reaches_name[i]
            location.Chainage = chainages[i]
            x_section = file.GetCrossSection(location, reaches_topoid[i])
            X_list, Z_list = [], []
            for j in range(x_section.BaseCrossSection.Points.Count):
                X_list.append(x_section.BaseCrossSection.Points.LstPoints[j].X)
                Z_list.append(x_section.BaseCrossSection.Points.LstPoints[j].Z)
            X_d = pd.Series(X_list, name='X:' + str(queries[i]))
            Z_d = pd.Series(Z_list, name='Z:' + str(queries[i]))

            df = pd.concat([df, X_d, Z_d], axis=1)

        return df

    def _get_data(self, file, queries, reaches_name, reaches_topoid, chainages):
        df = self._get_values(file, queries, reaches_name, reaches_topoid, chainages)
        return df

    def find_items(self, file, queries, chainage_tolerance=0.1):

        reaches_name = []
        reaches_topoid = []
        chainages = []

        reaches = file.GetReachTopoIdEnumerable()
        for query in queries:
            reach_name = -999
            reach_topoid = -999
            chainage = -999
            for reach in reaches:
                if reach.ReachId.strip() == query.reach_name.strip():
                    if reach.TopoId.strip() == query.topo_id.strip():
                        for cross_section in reach.GetChainageSortedCrossSections():
                            chainage_diff = float(cross_section.Key) - query.chainage
                            is_correct_chainage = abs(chainage_diff) < chainage_tolerance
                            if is_correct_chainage:
                                chainage = float(cross_section.Key)
                                break
                        reach_topoid = reach.TopoId.strip()
                    reach_name = reach.ReachId.strip()

            reaches_name.append(reach_name)
            reaches_topoid.append(reach_topoid)
            chainages.append(chainage)

            if -999 in reaches_name:
                raise Exception("Reach Not Found: {}".format(query.reach_name.strip()))
            if -999 in reaches_topoid:
                raise Exception(
                    "Topo-ID Not Found {} in Reach: {}".format(query.topo_id.strip(), query.reach_name.strip()))
            if -999 in chainages:
                raise Exception(
                    "Chainage {} Not Found in Reach/Topo-ID: {}/{}".format(query.chainage, query.reach_name.strip(),
                                                                           query.topo_id.strip()))

        return reaches_name, reaches_topoid, chainages

    def read(self, file_path, queries):

        file = self.__read(file_path)
        reaches_name, reaches_topoid, chainages = self.find_items(file, queries)
        df = self._get_data(file, queries, reaches_name, reaches_topoid, chainages)

        return df


class QueryData:
    """A query object that declares what data should be
    extracted from a .xns11 file.
    
    Parameters
    ----------
    topo_id: str
        Topo ID, must be passed
    reach_name: str, optional
        Reach name, consider all the reaches if None
    chainage: float, optional
        chainage, considers all the chainages if None
    
    Examples
    --------
    `QueryData('topoid1', 'reach1', 10)` is a valid query.
    `QueryData('topoid1', 'reach1')` requests all the points
    for `topoid1` of `reach1`.
    `QueryData('topoid1')` requests all the points for `topid1` 
    of the file.
    """

    def __init__(self, topo_id, reach_name=None, chainage=None):
        self._topo_id = topo_id
        self._reach_name = reach_name
        self._chainage = chainage
        self._validate()

    def _validate(self):
        tp = self.topo_id
        rn = self.reach_name
        c = self.chainage
        if not isinstance(tp, str):
            raise TypeError("topo_id must be a string.")
        if rn is not None and not isinstance(rn, str):
            raise TypeError("reach_name must be either None or a string.")
        if c is not None and not isinstance(c, (int, float)):
            raise TypeError("chainage must be either None or a number.")
        if rn is None and c is not None:
            raise ValueError("chainage cannot be set if reach_name is None.")

    @property
    def topo_id(self):
        return self._topo_id

    @property
    def reach_name(self):
        return self._reach_name

    @property
    def chainage(self):
        return self._chainage

    def __repr__(self):
        return (
            f"QueryData(topo_id='{self.topo_id}', "
            f"reach_name='{self.reach_name}', "
            f"chainage={self.chainage})"
        )
