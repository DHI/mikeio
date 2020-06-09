# Developed by MMF - June/2020 - Brazil
import clr
import os.path
import sys
import pandas as pd

#sys.path.append(r"C:/Program Files (x86)/DHI/2019/bin/") 

clr.AddReference("DHI.Mike1D.CrossSectionModule")
from DHI.Mike1D.CrossSectionModule import (
    CrossSectionDataFactory,
)

clr.AddReference("DHI.Mike1D.Generic")
from DHI.Mike1D.Generic import (
    Connection,
    Diagnostics,
    Location
)

clr.AddReference("System")

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
            X_d = pd.Series(X_list, name = 'X:' + str(queries[i]))
            Z_d = pd.Series(Z_list, name = 'Z:' + str(queries[i]))

            df = pd.concat([df, X_d, Z_d], axis = 1)

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
                if (reach.ReachId.strip() == query.BranchName.strip()):
                    if (reach.TopoId.strip() == query.TopoId.strip()):
                        for cross_section in reach.GetChainageSortedCrossSections():
                            chainage_diff = float(cross_section.Key) - query.Chainage
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
                raise Exception("Reach Not Found: {}".format(query.BranchName.strip()))
            if -999 in reaches_topoid:
                raise Exception("Topo-ID Not Found {} in Reach: {}".format(query.TopoId.strip(), query.BranchName.strip()))
            if -999 in chainages:
                raise Exception("Chainage {} Not Found in Reach/Topo-ID: {}/{}".format(query.Chainage, query.BranchName.strip() ,query.TopoId.strip()))

    
        return reaches_name, reaches_topoid, chainages

    def read(self, file_path, queries):
            
        file = self.__read(file_path)
        reaches_name, reaches_topoid, chainages = self.find_items(file, queries)
        df = self._get_data(file, queries, reaches_name, reaches_topoid, chainages)

        return df

class ExtractionPoint:

    def BranchName(BranchName):
        """
            Name of the Branch
        """
        return BranchName

    def TopoId(TopoId):
        """
            Name of the TopoId
        """
        return TopoId

    def Chainage(Chainage):
        """
            Chainage number along branch
        """
        return Chainage

    def __str__(self):
        return f"{self.BranchName} {self.TopoId} {self.Chainage}"
