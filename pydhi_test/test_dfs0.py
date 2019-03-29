from unittest import TestCase
import numpy as np
import pandas as pd
import datetime
from pydhi import dfs0 as dfs0
from pydhi import dfs_util as dfs_util

class test_dfs0(TestCase):

    def test_create_equidistant_calendar(self):

        dfs0file = r'C:\test\randomEQC.dfs0'
        data = np.random.random([1000, 2])
        data[2, :] = np.nan
        start_time = datetime.datetime(2017, 1, 1)
        timeseries_unit = 1402
        title = 'Hello Test'
        names = ['VarFun01', 'NotFun']
        variable_type = [100000, 100000]
        unit = [1000, 1000]
        data_value_type = [0, 1]
        dt = 5
        dfs = dfs0.dfs0()
        dfs.create_equidistant_calendar(dfs0file=dfs0file, data=data, start_time=start_time,
                                        timeseries_unit=timeseries_unit, dt=dt, names=names,
                                        title=title, variable_type=variable_type, unit=unit,
                                        data_value_type=data_value_type)


    def test_create_non_equidistant_calendar(self):
        dfs0file = r'C:\test\randomEQC.dfs0'
        data = np.random.random([1000, 2])
        data[2, :] = np.nan
        start_time = datetime.datetime(2017, 1, 1)
        time_vector = []
        for i in range(1000):
            time_vector.append( start_time + datetime.timedelta(hours=i*0.1) )
        title = 'Hello Test'
        names = ['VarFun01', 'NotFun']
        variable_type = [100000, 100000]
        unit = [1000, 1000]
        data_value_type = [0, 1]


        dfs = dfs0.dfs0()
        dfs.create_non_equidistant_calendar(dfs0file=dfs0file, data=data, time_vector=time_vector,
                                            names=names, title=title, variable_type=variable_type, unit=unit,
                                            data_value_type=data_value_type)


    def test_read_dfs0_to_pandas(self):

        dfs0file = r'C:\test\randomEQC.dfs0'

        dfs = dfs0.dfs0()
        pd = dfs.read_to_pandas(dfs0file)

        self.assertEqual(np.isnan(pd[pd.columns[0]][2]), True)


    def test_read_dfs0_to_matrix(self):
        dfs0file = r'C:\test\randomEQC.dfs0'

        dfs = dfs0.dfs0()
        mat = dfs.read(dfs0file, indices=[0])[0]

        self.assertEqual(np.isnan(mat[2, 0]), True)
